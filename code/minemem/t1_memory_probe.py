"""Run the 20-trajectory memory sensitivity gate and cached waypoint probes."""

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from t1_spatial import GeometricSpatialMemory, SpatialMemory, spatial_read_plan


DISTANCES = (10, 30, 60, 100)
TASKS = ("waypoint_direction", "waypoint_distance", "gold_direction")
INTERVENTIONS = ("rotate_pose", "shuffle_visual", "zero_visual")


def direction_and_distance(pose, target):
    x, z, yaw = pose
    heading = math.degrees(math.atan2(x-target[0], target[1]-z))
    distance = math.dist((x, z), target)
    return int(((heading-yaw+22.5) % 360) // 45), int(distance >= 5) + int(distance >= 15)


def teacher_labels(meta, frames):
    scene = meta["scene"]
    targets = scene["teacher_targets"]
    pose = meta["pose"]
    sampled = set(frames)
    waypoint = 0
    labels = []
    for index in range(meta["outbound_frames"], frames[-1]+1):
        while waypoint+1 < len(targets) and math.dist(pose[index][:2], targets[waypoint]) <= 0.45:
            waypoint += 1
        if index in sampled:
            direction, distance = direction_and_distance(pose[index], targets[waypoint])
            gold, _ = direction_and_distance(pose[index], scene["landmark"])
            labels.append((direction, distance, gold))
    return torch.tensor(labels, dtype=torch.long)


def plan_batch(history, current, side):
    indices, geometry, bias = zip(*(spatial_read_plan(history, pose, side) for pose in current), strict=True)
    return tuple(torch.from_numpy(np.stack(part)) for part in (indices, geometry, bias))


def feature_statistics(episodes):
    count = 0
    total = squares = None
    for episode in episodes:
        feature = episode["grid"].float().flatten(0, 1)
        if total is None:
            total = torch.zeros(feature.shape[-1], dtype=torch.float64)
            squares = torch.zeros_like(total)
        count += len(feature)
        total += feature.sum(dim=0, dtype=torch.float64)
        squares += feature.square().sum(dim=0, dtype=torch.float64)
    center = total/count
    scale = (squares/count-center.square()).clamp_min(1e-4).sqrt()
    return center.float(), scale.float()


def make_examples(episode, mean, std, side):
    count = len(episode["history"])
    poses = np.asarray(episode["meta"]["pose"], dtype=np.float32)
    history, current = poses[episode["history"]], poses[episode["current"]]
    feature = ((episode["grid"].float()-mean)/std).clamp(-8, 8)
    seed = episode["row"]["scene_seed"]
    rng = np.random.default_rng(seed)
    return dict(row=episode["row"], feature=feature, history=history, current=current,
                history_length=count, labels=teacher_labels(episode["meta"], episode["current"]),
                plan=plan_batch(history, current, side), permutation=torch.from_numpy(rng.permutation(count)),
                legacy_mean=episode["legacy_mean"][:count])


class SpatialDecoder(nn.Module):
    def __init__(self, memory):
        super().__init__()
        self.memory = memory
        self.memory.to_llm.requires_grad_(False)
        self.waypoint = nn.Linear(memory.width*3, 8)
        self.distance = nn.Linear(memory.width*3, 3)
        self.gold = nn.Linear(memory.width, 8)

    def forward(self, case, visual=None, plan=None):
        indices, geometry, ranks = case["plan"] if plan is None else plan
        if visual is None:
            visual = case["feature"][:case["history_length"]]
        encoded = self.memory.encode_visual(visual)
        memory = self.memory.read_projected(encoded[indices], geometry, ranks)
        local = memory[:, :3].flatten(1)
        return self.waypoint(local), self.distance(local), self.gold(memory[:, 0])


class CurrentDecoder(nn.Module):
    def __init__(self, dimension, patches):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dimension, 16, bias=False), nn.LayerNorm(16))
        self.head = nn.Linear(patches*16, 19)

    def forward(self, case):
        current = case["feature"][case["history_length"]:]
        return self.head(self.project(current).flatten(1)).split((8, 3, 8), dim=-1)


def rotated_history(history, current):
    result = history.copy()
    result[:, 0] = current[0] - (history[:, 1]-current[1])
    result[:, 1] = current[1] + (history[:, 0]-current[0])
    result[:, 2] = (history[:, 2]+90) % 360
    return result


@torch.no_grad()
def sensitivity(memory, cases, side, legacy=None):
    rows = []
    for case in cases:
        current = case["current"][0]
        history = case["history"]
        original = tuple(part[0] for part in case["plan"])
        rotated = tuple(torch.from_numpy(part) for part in spatial_read_plan(rotated_history(history, current),
                                                                             current, side))
        visual = case["feature"][:case["history_length"]]

        def prefix(features, plan):
            indices, geometry, bias = plan
            return memory(features[indices], geometry, bias)

        reference = prefix(visual, original)
        alternatives = (prefix(visual, rotated), prefix(visual[case["permutation"]], original),
                        prefix(torch.zeros_like(visual), original))
        values = {name: float((other-reference).norm()/reference.norm())
                  for name, other in zip(INTERVENTIONS, alternatives, strict=True)}
        row = dict(seed=case["row"]["scene_seed"], distance=case["row"]["group"],
                   prefix_norm=float(reference.norm()), token_norm=float(reference.norm(dim=-1).mean()), **values)
        if legacy is not None:
            positions = np.vstack((history, current))
            old_reference = legacy(case["legacy_mean"], positions)[0]
            row.update(legacy_prefix_norm=float(old_reference.norm()),
                       legacy_token_norm=float(old_reference.norm(dim=-1).mean()))
        rows.append(row)
    if len(rows) != 20 or len({row["seed"] for row in rows}) != 20:
        raise ValueError("Sensitivity gate requires the twenty disjoint validation trajectories")
    summary = {name: dict(minimum=min(row[name] for row in rows),
                          mean=sum(row[name] for row in rows)/len(rows),
                          passed=sum(row[name] >= .10 for row in rows)) for name in INTERVENTIONS}
    return dict(passed=all(value["passed"] == 20 for value in summary.values()),
                thresholds=summary, rows=rows)


@torch.no_grad()
def evaluate(spatial, no_memory, cases, side):
    counts = Counter()
    predictions = []
    for case in cases:
        meta = case["row"]
        labels = case["labels"]
        perm = case["permutation"]
        history = case["history"]
        # Break the pose-to-frame assignment. This corrupts the route geometry as well as appearance alignment.
        shuffled_plan = plan_batch(history[perm.numpy()], case["current"], side)
        outputs = {
            "spatial": spatial(case),
            "current_frame": no_memory(case),
            "shuffled_pose": spatial(case, plan=shuffled_plan),
            "shuffled_visual": spatial(case, visual=case["feature"][:len(history)][perm]),
        }
        for name, logits in outputs.items():
            guessed = torch.stack([score.argmax(-1) for score in logits], dim=-1)
            matched = guessed.eq(labels)
            for distance in (meta["group"], "all"):
                counts[name, distance, "samples"] += len(labels)
                counts[name, distance, "starts"] += 1
                for k, task in enumerate(TASKS):
                    counts[name, distance, task] += int(matched[:, k].sum())
                    counts[name, distance, f"{task}_at_start"] += int(matched[0, k])
            for offset, (truth, result) in enumerate(zip(labels.tolist(), guessed.tolist(), strict=True)):
                predictions.append(dict(seed=meta["scene_seed"], world=meta["world_seed"], distance=meta["group"],
                                        sample=offset, model=name, target=truth, prediction=result))
    summary = {}
    for name in outputs:
        summary[name] = {}
        for distance in (*DISTANCES, "all"):
            samples = counts[name, distance, "samples"]
            starts = counts[name, distance, "starts"]
            summary[name][distance] = dict(samples=samples, starts=starts,
                                          **{task: counts[name, distance, task]/samples for task in TASKS},
                                          **{f"{task}_at_start": counts[name, distance, f"{task}_at_start"]/starts
                                             for task in TASKS})
    return summary, predictions


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def run(args):
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.cache / "cache_config.json").read_text())
    embeddings = json.loads((args.cache / "embedding_norms.json").read_text())
    raw = [torch.load(args.cache / f"{row['scene_seed']}.pt", map_location="cpu", weights_only=True)
           for row in info["episodes"]]
    mean, std = feature_statistics([episode for episode in raw if episode["row"]["split"] == "train"])
    examples = [make_examples(episode, mean, std, args.grid_side) for episode in raw]
    del raw
    train, validation, test = ([example for example in examples if example["row"]["split"] == split]
                               for split in ("train", "validation", "test"))
    dim = len(mean)
    patch_side = info["spatial_pool_side"]
    memory = GeometricSpatialMemory(dim, embeddings["dimension"], embeddings["ordinary_mean"],
                                    patch_side=patch_side, grid_side=args.grid_side)
    spatial = SpatialDecoder(memory)
    current = CurrentDecoder(dim, patch_side**2)
    legacy = None
    if args.legacy_weights:
        weights = load_file(args.legacy_weights)
        legacy = SpatialMemory(dim, embeddings["dimension"], width=weights["write.0.weight"].shape[0])
        legacy.load_state_dict(weights)
        legacy.eval()
    initial = sensitivity(memory, validation, args.grid_side, legacy)
    save_json(args.output / "sensitivity_initial.json", initial)
    save_json(args.output / "embedding_norms.json", embeddings)
    print("initial sensitivity", json.dumps(initial["thresholds"]), flush=True)
    if not initial["passed"]:
        raise RuntimeError("Initial memory sensitivity gate failed")

    parameters = [parameter for model in (spatial, current) for parameter in model.parameters()
                  if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.003)
    order = list(range(len(train)))
    rng = random.Random(args.seed)
    with (args.output / "training.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "spatial_loss", "current_frame_loss"))
        for epoch in range(1, args.epochs+1):
            rng.shuffle(order)
            totals = [0.0, 0.0]
            for index in order:
                case = train[index]
                visual = case["feature"][:case["history_length"]]
                if rng.random() < .35:
                    visual = torch.zeros_like(visual)
                losses = [sum(F.cross_entropy(logits, case["labels"][:, j])
                              for j, logits in enumerate(prediction))
                          for prediction in (spatial(case, visual), current(case))]
                optimizer.zero_grad(set_to_none=True)
                sum(losses).backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                for j, loss in enumerate(losses):
                    totals[j] += float(loss.detach())
            writer.writerow((epoch, *(value/len(train) for value in totals)))
            stream.flush()
            if epoch == 1 or epoch % 5 == 0:
                print(f"epoch={epoch} mean_loss={','.join(f'{value/len(train):.4f}' for value in totals)}", flush=True)
    spatial.eval()
    current.eval()
    trained = sensitivity(memory, validation, args.grid_side)
    save_json(args.output / "sensitivity_trained.json", trained)
    results = {}
    for label, cases in (("validation", validation), ("test", test)):
        results[label], predictions = evaluate(spatial, current, cases, args.grid_side)
        save_json(args.output / f"{label}_predictions.json", predictions)
    spatial_result = results["validation"]["spatial"]["all"]
    passed = all(spatial_result[task] >= .80 for task in ("waypoint_direction", "gold_direction"))
    summary = dict(sensitivity_passed=initial["passed"] and trained["passed"], decoding_passed=passed,
                   proceed=initial["passed"] and trained["passed"] and passed, grid_cells=args.grid_side,
                   grid_width_blocks=args.grid_side*2, train_episodes=len(train),
                   validation_episodes=len(validation), test_episodes=len(test), epochs=args.epochs,
                   spatial_trainable=sum(p.numel() for p in spatial.parameters() if p.requires_grad),
                   current_trainable=sum(p.numel() for p in current.parameters() if p.requires_grad),
                   direction_bins=8, distance_thresholds_blocks=[5, 15], results=results)
    save_json(args.output / "summary.json", summary)
    torch.save(dict(model=spatial.state_dict(), mean=mean, std=std, grid_side=args.grid_side,
                    patch_side=patch_side, vision_dim=dim, embedding_dim=embeddings["dimension"],
                    embedding_norm=embeddings["ordinary_mean"]), args.output / "waypoint-memory.pt")
    torch.save(dict(model=current.state_dict(), mean=mean, std=std), args.output / "current-frame.pt")
    print("trained sensitivity", json.dumps(trained["thresholds"]), flush=True)
    print("validation", json.dumps(results["validation"]), flush=True)
    if not summary["proceed"]:
        raise RuntimeError("Memory gate blocked the next phase")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-weights", type=Path)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--grid-side", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--learning-rate", type=float, default=.002)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("Training epochs and learning rate must be positive")
    run(args)
