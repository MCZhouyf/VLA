"""Probe what visual and geometric memory contribute on unseen T1 returns."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from t1_spatial import retained_cells
from t1_steering_probe import steering_action
from t1_storage import retry_read
from t1_train import case_examples, load_images, mask_history
from t1_trained_eval import Controller


def teacher_inputs(controller, row, cache, offsets):
    start = row["outbound_frames"]
    stop = start + max(offsets) + 1
    indices = list(range(start, stop))
    seed = controller.settings["seed"]
    random.Random(seed + row["scene_seed"]).shuffle(indices)
    examples = case_examples(
        controller.policy, dict(row, selected_frames=stop), seed, controller.settings["resolution"],
        controller.cache_vision, cache,
    )
    return {index-start: example for index, example in zip(indices, examples, strict=True)
            if index-start in offsets}


def sparse_teacher_input(controller, archive, metadata, action_index, stride):
    policy = controller.policy
    recent_start = action_index - policy.history
    indices = [*range(0, recent_start, stride), *range(recent_start, action_index + 1)]
    frames = []
    for begin in range(0, len(indices), 3):
        images = load_images(archive, indices[begin:begin + 3], policy.images,
                             controller.settings["resolution"])
        frames.extend(policy.encode_frames(images))
    poses = [metadata["pose"][index] for index in indices]
    actions = metadata.get("executed_action", metadata["action"])[recent_start:action_index]
    dwell = metadata["scene"]["outbound_dwell_steps"]
    return (metadata["instruction"], frames[-(policy.history + 1):], actions, metadata["action"][action_index],
            policy.pooled_images(frames[:-1]), poses), sum(index < dwell for index in indices[:-1])


def condition_inputs(features, poses, seed):
    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    x, z, _ = poses[-1]
    rotated = [(x - (old_z-z), z + (old_x-x), (yaw+90) % 360) for old_x, old_z, yaw in poses[:-1]]
    return {
        "real_memory": (features, poses),
        "shuffled_visual_content": (features[indices], poses),
        "zero_visual_content": (torch.zeros_like(features), poses),
        "rotated_historical_poses": (features, [*rotated, poses[-1]]),
    }


def score(policy, ground_truth, prediction, logits, targets):
    action, _, complete, ids = prediction
    direction, yaw, forward = steering_action(policy.actions, action)
    log_probs = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return dict(
        direction=direction, yaw_bin=yaw, forward=forward, complete=bool(complete),
        generated_token_ids=ids, exact_teacher=ids == ground_truth,
        teacher_log_probability=float(log_probs.sum()),
    )


def compare(policy, example, seed, wider_grid_side, grid_only, initial_frames):
    instruction, frames, actions, target, features, poses = mask_history(example, policy.actions.null_token())
    conditions = ({"real_memory": (features, poses)} if grid_only
                  else condition_inputs(features, poses, seed))
    memory = policy.memory
    reference_prefix = memory(features, poses).float()
    scale = reference_prefix.norm().clamp_min(1e-12)
    requests = [(instruction, frames, actions, packed, positions) for packed, positions in conditions.values()]
    inputs = [(i, f, a, target, v, p) for i, f, a, v, p in requests]
    teacher = policy.action_ids(target)
    predictions = policy.predict_batch(requests, return_ids=True)
    scores = {}
    for name, prediction, (logits, targets) in zip(
            conditions, predictions, policy.batched_action_logits(inputs), strict=True):
        prefix = memory(*conditions[name]).float()
        scores[name] = score(policy, teacher, prediction, logits, targets)
        scores[name]["relative_prefix_delta"] = float((prefix-reference_prefix).norm()/scale)
    if not grid_only:
        policy.memory = None
        try:
            request = (instruction, frames, actions)
            prediction = policy.predict(*request, return_ids=True)
            logits, targets = policy.action_logits(*request, target)
            scores["no_prefix"] = score(policy, teacher, prediction, logits, targets)
        finally:
            policy.memory = memory
    if wider_grid_side:
        original = memory.grid_side
        positions = np.asarray(poses, dtype=np.float32)
        _, _, old_keep = retained_cells(positions, original)
        _, _, new_keep = retained_cells(positions, wider_grid_side)
        scores["real_memory"].update(retained_frames=int(old_keep.sum()),
                                     retained_initial_frames=int(old_keep[:initial_frames].sum()))
        memory.grid_side = wider_grid_side
        try:
            prediction, = policy.predict_batch([(instruction, frames, actions, features, poses)], return_ids=True)
            (logits, targets), = policy.batched_action_logits([(instruction, frames, actions, target, features, poses)])
            wider = score(policy, teacher, prediction, logits, targets)
            wider.update(relative_prefix_delta=float((memory(features, poses).float()-reference_prefix).norm()/scale),
                         retained_frames=int(new_keep.sum()),
                         retained_initial_frames=int(new_keep[:initial_frames].sum()))
            scores["wider_grid"] = wider
        finally:
            memory.grid_side = original
    return teacher, scores


@torch.inference_mode()
def run(args):
    controller = Controller(args.checkpoint, args.training_run, device=args.device)
    policy = controller.policy
    if policy.memory is None or controller.settings.get("evaluation_history") != "placeholder":
        raise ValueError("The probe requires a repaired spatial checkpoint and placeholder history")
    dataset = retry_read(args.dataset.read_bytes)
    digest = hashlib.sha256(dataset).hexdigest()
    if digest != controller.settings["dataset_sha256"]:
        raise ValueError("The checkpoint and held-out index lineage differ")
    rows = [row for line in dataset.splitlines()
            if (row := json.loads(line))["split"] == "validation" and row["group"] == args.distance
            and row["selected_labels"] > max(args.offsets)]
    rows.sort(key=lambda row: (row["outbound_frames"], row["scene_seed"]))
    distinct_worlds = {}
    for row in rows:
        distinct_worlds.setdefault(row["world_seed"], row)
    selected = list(distinct_worlds.values())[:args.episodes]
    if len(selected) != args.episodes:
        raise ValueError("Insufficient held-out worlds for the requested comparison")
    results = []
    for row in selected:
        cache_path = args.archive_cache / Path(row["path"]).name
        print(f"encoding seed={row['scene_seed']} outbound={row['outbound_frames']} stride={args.history_stride}",
              flush=True)
        with retry_read(ZipFile, cache_path) as stream:
            metadata = json.loads(retry_read(stream.read, "episode.json"))
            if args.history_stride == 1:
                dwell = metadata["scene"]["outbound_dwell_steps"]
                examples = {offset: (example, dwell) for offset, example in teacher_inputs(
                    controller, row, args.archive_cache, args.offsets).items()}
            else:
                examples = {offset: sparse_teacher_input(controller, stream, metadata,
                            row["outbound_frames"] + offset, args.history_stride) for offset in args.offsets}
        for offset, (example, initial_frames) in sorted(examples.items()):
            position = example[-1][-1][:2]
            teacher, scores = compare(policy, example, f"{row['scene_seed']}:{offset}:content",
                                      args.wider_grid_side, args.grid_only, initial_frames)
            results.append(dict(scene_seed=row["scene_seed"], world_seed=row["world_seed"], offset=offset,
                                true_horizontal_distance=math.dist(position, metadata["scene"]["landmark"]),
                                history_frames=len(example[-2]), teacher_token_ids=teacher, conditions=scores))
            print(f"scored seed={row['scene_seed']} offset={offset}", flush=True)
    changes = Counter()
    counts = Counter()
    for row in results:
        reference = row["conditions"]["real_memory"]
        for name, score_row in row["conditions"].items():
            changes[name] += score_row["generated_token_ids"] != reference["generated_token_ids"]
            counts[name] += score_row["exact_teacher"]
    report = dict(distance=args.distance, device=args.device, dtype=str(policy.base.dtype),
                  dataset_sha256=digest, selection="shortest outbound histories from distinct validation worlds",
                  wider_grid_side=args.wider_grid_side, history_stride=args.history_stride,
                  tested_inputs=len(results), changed_outputs_vs_real_memory=changes,
                  exact_teacher_actions=counts, rows=results)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(tested_inputs=len(results), changes=changes, exact_teacher=counts)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "training-run", "dataset", "archive-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--distance", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--offsets", type=int, nargs="+", default=(0, 24))
    parser.add_argument("--wider-grid-side", type=int, choices=(64, 128))
    parser.add_argument("--grid-only", action="store_true")
    parser.add_argument("--history-stride", type=int, default=1,
                        help="Sample historical keyframes at this stride; always keep recent model frames")
    args = parser.parse_args()
    if (args.episodes <= 0 or args.history_stride <= 0 or min(args.offsets) < 0
            or len(set(args.offsets)) != len(args.offsets)):
        parser.error("Provide positive episode count and history stride, and distinct nonnegative offsets")
    if args.grid_only and args.wider_grid_side is None:
        parser.error("Grid-only comparison requires an enlarged grid")
    run(args)


if __name__ == "__main__":
    main()
