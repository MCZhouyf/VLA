"""Measure free-generation steering on held-out visible-goal actions."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

import torch
from minestudio.utils.vpt_lib.actions import Buttons

from t1_storage import retry_read
from t1_train import case_examples, mask_history
from t1_trained_eval import Controller


CLASSES = ("forward", "left", "right")
NAVIGATION_TOKENS = {
    "forward": (178, 191, 219, 240, 179),
    "left": (178, 219, 230, 179),
    "right": (178, 219, 250, 179),
}


def steering_action(codec, agent):
    controls = codec.action_mapper.to_factored(agent)
    yaw_bin = int(controls["camera"][1])
    center = codec.n_camera_bins // 2
    forward = bool(controls["buttons"][Buttons.ALL.index("forward")])
    if yaw_bin < center:
        return "left", yaw_bin, forward
    if yaw_bin > center:
        return "right", yaw_bin, forward
    return ("forward" if forward else "other"), yaw_bin, forward


def choose_actions(rows, archive_cache, classify, per_class, seed):
    candidates = defaultdict(list)
    for row in rows:
        path = archive_cache / Path(row["path"]).name
        with retry_read(ZipFile, path) as archive:
            labels = json.loads(retry_read(archive.read, "episode.json"))["action"]
        for frame in range(row["outbound_frames"], row["selected_frames"]):
            direction, _, _ = classify(labels[frame])
            if direction in CLASSES:
                candidates[direction].append((row["scene_seed"], frame))
    rng = random.Random(seed)
    return {sample for direction in CLASSES for sample in rng.sample(candidates[direction], per_class)}


def validation_examples(rows, selected, controller, archive_cache):
    policy = controller.policy
    seed = controller.settings["seed"]
    for row in rows:
        scene = row["scene_seed"]
        frames = {index for candidate, index in selected if candidate == scene}
        if not frames:
            continue
        stop = max(frames) + 1
        indices = list(range(row["outbound_frames"], stop))
        random.Random(seed + scene).shuffle(indices)
        examples = case_examples(policy, dict(row, selected_frames=stop), seed,
                                 controller.settings["resolution"], controller.cache_vision, archive_cache)
        for index, example in zip(indices, examples, strict=True):
            if index in frames:
                yield scene, index, mask_history(example, policy.actions.null_token())


def score_complete_navigation_actions(policy, request):
    instruction, frames, actions, features, poses = request
    labels = ["".join(f"<|reserved_special_token_{number}|>" for number in NAVIGATION_TOKENS[name])
              for name in CLASSES]
    inputs = [(instruction, frames, actions, label, features, poses) for label in labels]
    results = {}
    branch = None
    for direction, (logits, targets) in zip(CLASSES, policy.batched_action_logits(inputs), strict=True):
        log_probs = logits.log_softmax(-1)
        scores = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        results[direction] = dict(log_probability=float(scores.sum()), token_log_probabilities=scores.tolist())
        if branch is None:
            begin = log_probs[1]
            branch = dict(forward=float(begin[policy.action_ids(labels[0])[1]].exp()),
                          no_forward_neutral_pitch=float(begin[policy.action_ids(labels[1])[1]].exp()))
    return dict(scores=results, first_branch_probabilities=branch,
                best=max(CLASSES, key=lambda direction: results[direction]["log_probability"]))


@torch.inference_mode()
def run(args):
    controller = Controller(args.checkpoint, args.training_run, device=args.device)
    if controller.settings.get("evaluation_history") != "placeholder":
        raise ValueError("This probe requires the repaired placeholder-history configuration")
    index = retry_read(args.dataset.read_bytes)
    digest = hashlib.sha256(index).hexdigest()
    if digest != controller.settings["dataset_sha256"]:
        raise ValueError("The probe requires the training run's frozen dataset index")
    rows = [row for line in index.splitlines() if (row := json.loads(line))["split"] == "validation"
            and row.get("group") == "visible" and row["selected_labels"] > 0]
    policy = controller.policy
    if args.compare_memory_read and policy.memory is None:
        raise ValueError("A memory-read comparison requires a spatial checkpoint")

    @lru_cache(maxsize=None)
    def classify(label):
        return steering_action(policy.actions, policy.actions.decode(policy.action_ids(label))[0])

    selected = choose_actions(rows, args.archive_cache, classify, args.per_class, args.seed)
    cases = validation_examples(rows, selected, controller, args.archive_cache)
    results = []
    while group := list(islice(cases, args.batch_size)):
        requests = [(instruction, frames, actions, features, poses)
                    for _, _, (instruction, frames, actions, _, features, poses) in group]
        predictions = policy.predict_batch(requests, return_ids=True)
        ablated = [None] * len(group)
        if args.compare_memory_read:
            memory = policy.memory
            policy.memory = None
            try:
                ablated = policy.predict_batch(requests, return_ids=True)
            finally:
                policy.memory = memory
        for (scene, index, example), request, (action, _, complete, ids), comparison in zip(
                group, requests, predictions, ablated, strict=True):
            expected, expected_yaw, expected_forward = classify(example[3])
            actual, yaw, forward = steering_action(policy.actions, action)
            entry = dict(scene_seed=scene, frame=index, expected=expected, expected_yaw_bin=expected_yaw,
                         expected_forward=expected_forward, predicted=actual if complete else "invalid",
                         predicted_yaw_bin=yaw, predicted_forward=forward,
                         raw_token_ids=ids, complete=bool(complete))
            if comparison is not None:
                without, _, valid, alternative_ids = comparison
                direction, alternative_yaw, alternative_forward = steering_action(policy.actions, without)
                entry["without_memory_read"] = dict(predicted=direction if valid else "invalid",
                                                     predicted_yaw_bin=alternative_yaw,
                                                     predicted_forward=alternative_forward,
                                                     raw_token_ids=alternative_ids)
            if args.score_complete_actions:
                entry["complete_action_comparison"] = score_complete_navigation_actions(policy, request)
            results.append(entry)
    totals = Counter(row["expected"] for row in results)
    if totals != dict.fromkeys(CLASSES, args.per_class):
        raise RuntimeError("Validation coverage differs from the balanced sample")
    confusion = {expected: dict(Counter(row["predicted"] for row in results if row["expected"] == expected))
                 for expected in CLASSES}
    scores = {expected: confusion[expected].get(expected, 0) / totals[expected] for expected in CLASSES}
    result = dict(dataset_sha256=digest, variant=controller.settings["variant"], selection_seed=args.seed,
                  device=args.device, dtype=str(policy.base.dtype), examples_per_class=args.per_class,
                  confusion=confusion, recalls=scores, rows=results)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(confusion=confusion, recalls=scores)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "training-run", "dataset", "archive-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--per-class", type=int, default=13)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--compare-memory-read", action="store_true")
    parser.add_argument("--score-complete-actions", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.per_class <= 0:
        parser.error("Batch size and per-class count must be positive")
    run(args)


if __name__ == "__main__":
    main()
