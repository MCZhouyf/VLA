"""Train and audit three small waypoint predictors on the same frozen T1 v3 cache."""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from t1_memory_probe import rotated_history
from t1_spatial import GeometricSpatialMemory, QUERIES, spatial_read_plan
from t1_storage import retry_read
from t1_waypoint import DIRECTION_BINS, relative_waypoint, teacher_waypoints, waypoint_labels


VARIANTS = ("no_memory", "long", "spatial")
MAX_WAYPOINT_DISTANCE = 16
GOAL_DISTANCE_SCALE = 5.0


def short_plan(current, rows, history_size):
    offsets = np.rint(np.linspace(history_size, 1, QUERIES)).astype(np.int64)
    selection = np.stack([np.repeat([rows[max(0, frame-offset)] for offset in offsets], 2)
                          for frame in current]).reshape(-1, QUERIES, 2)
    geometry = np.zeros((*selection.shape, 9), dtype=np.float32)
    geometry[..., 6] = np.log1p(offsets)[None, :, None] / 8
    return selection, geometry, np.zeros(selection.shape, dtype=np.float32)


def mapped_plan(poses, current, side, history_rows):
    selections, geometry, bias = zip(*(spatial_read_plan(poses, point, side) for point in current), strict=True)
    return history_rows[np.stack(selections)], np.stack(geometry), np.stack(bias)


def load_case(record, side, device):
    row, meta = record["row"], record["meta"]
    source = {frame: index for index, frame in enumerate(record["frame_indices"])}
    frames = record["labels"]
    poses = np.asarray(meta["pose"], dtype=np.float32)
    current = poses[frames]
    route = teacher_waypoints(meta)
    targets = [route[frame-meta["outbound_frames"]] for frame in frames]
    waypoints = [waypoint_labels(pose, target) for pose, target in zip(current, targets, strict=True)]
    if any(distance > MAX_WAYPOINT_DISTANCE for _, distance in waypoints):
        raise ValueError(f"Waypoint distance out of range for {row['scene_seed']}")
    scene = meta["scene"]
    goals = [relative_waypoint(pose, scene["landmark"]) for pose in current]
    classes = [(direction, distance, round(angle/10) % DIRECTION_BINS)
               for (direction, distance), (angle, _) in zip(waypoints, goals, strict=True)]
    ranges = [math.log1p(distance)/GOAL_DISTANCE_SCALE for _, distance in goals]
    historical = poses[record["history"]]
    hist_rows = np.asarray([source[index] for index in record["history"]])
    plans = {"no_memory": short_plan(frames, source, 2), "long": short_plan(frames, source, 8),
             "spatial": mapped_plan(historical, current, side, hist_rows)}
    probes = {}
    if row["role"] == "validation":
        shuffled = np.random.default_rng(row["scene_seed"]).permutation(len(hist_rows))
        idx, geometry, bias = plans["spatial"]
        substitution = dict(zip(hist_rows, hist_rows[shuffled], strict=True))
        probes["shuffle_visual"] = np.vectorize(substitution.__getitem__, otypes=[np.int64])(idx), geometry, bias
        rotations = [spatial_read_plan(rotated_history(historical, point), point, side) for point in current]
        idx, geo, rank = map(np.stack, zip(*rotations, strict=True))
        probes["rotate_pose"] = hist_rows[idx], geo, rank
    def tensors(plan):
        return tuple(torch.from_numpy(part).to(device) for part in plan)
    return dict(row=row, scene=scene, grid=record["grid"].to(device),
                current=torch.tensor([source[f] for f in frames], device=device),
                classes=torch.tensor(classes, dtype=torch.long, device=device),
                range=torch.tensor(ranges, device=device), pose=current,
                candidate=int(scene.get("goal_slot",2)),
                plans={key:tensors(value) for key,value in plans.items()},
                probes={key:tensors(value) for key,value in probes.items()})


class Predictor(nn.Module):
    def __init__(self, initial, condition_candidate=False):
        super().__init__()
        self.reader = GeometricSpatialMemory(initial["vision_dim"], initial["embedding_dim"],
                                             initial["embedding_norm"], initial["patch_side"],
                                             grid_side=initial["grid_side"])
        self.reader.load_state_dict({key.removeprefix("memory."): value for key,value in initial["model"].items()
                                     if key.startswith("memory.")}, strict=True)
        self.reader.to_llm.requires_grad_(False)
        self.condition_candidate = condition_candidate
        width = self.reader.width
        self.head = nn.Sequential(nn.Linear(QUERIES*width + width-32 + 3*condition_candidate, 128),
                                  nn.LayerNorm(128), nn.GELU(),
                                  nn.Linear(128, 2*DIRECTION_BINS+MAX_WAYPOINT_DISTANCE+2))
        self.register_buffer("mean", initial["mean"])
        self.register_buffer("std", initial["std"])

    def forward(self, case, variant, intervention=None):
        grid = ((case["grid"].float()-self.mean)/self.std).clamp(-8, 8)
        encoded = self.reader.encode_visual(grid)
        indices, geometry, bias = case["probes"].get(intervention, case["plans"][variant])
        appearance = encoded[indices]
        if intervention == "zero_visual":
            shape = (1, grid.shape[-2], grid.shape[-1])
            appearance = self.reader.encode_visual(grid.new_zeros(shape)).expand_as(appearance)
        values = self.reader.read_projected(appearance, geometry, bias)
        if intervention == "remove_prefix":
            values = torch.zeros_like(values)
        features = torch.cat((values.flatten(1), encoded[case["current"]]), dim=-1)
        if self.condition_candidate:
            goal = features.new_zeros(3)
            goal[case["candidate"]] = 1
            features = torch.cat((features,goal.expand(len(features),-1)),dim=-1)
        return self.head(features).split((DIRECTION_BINS, MAX_WAYPOINT_DISTANCE+1, DIRECTION_BINS, 1), dim=-1)


def objective(logits, case):
    return sum(F.cross_entropy(logit, case["classes"][:, i]) for i,logit in enumerate(logits[:3])) + (
        F.smooth_l1_loss(logits[3].flatten(), case["range"]))


def candidate_prediction(case, direction, distance):
    candidates = case["scene"]["candidate_positions"]
    predicted = []
    for pose, course, radius in zip(case["pose"], direction, distance, strict=True):
        yaw = math.radians(pose[2]+course*10)
        estimate = (pose[0]-radius*math.sin(yaw), pose[1]+radius*math.cos(yaw))
        predicted.append(min(range(2), key=lambda target: math.dist(estimate, candidates[target])))
    return predicted


@torch.inference_mode()
def evaluate(models, cases, interventions):
    stats = defaultdict(Counter)
    detailed = []
    for case in cases:
        row = case["row"]
        outputs = {variant:model(case, variant) for variant,model in models.items()}
        if interventions:
            model = models["spatial"]
            outputs.update({name:model(case, "spatial", name)
                            for name in ("shuffle_visual", "zero_visual", "rotate_pose", "remove_prefix")})
        labels = case["classes"].cpu().tolist()
        for name,logits in outputs.items():
            predicted = torch.stack([value.argmax(-1) for value in logits[:3]], dim=-1).cpu().tolist()
            distance = torch.expm1(logits[3].flatten().clamp(0, 2)*GOAL_DISTANCE_SCALE).cpu().tolist()
            selected = candidate_prediction(case, [item[-1] for item in predicted], distance)
            for sample,(truth,guess) in enumerate(zip(labels,predicted,strict=True)):
                slot_ok = selected[sample] == case["scene"]["goal_slot"]
                keys = [(str(row["group"]), "all"), ("all", "all")]
                keys += [(str(row["group"]), f"candidate_{'correct' if slot_ok else 'wrong'}")]
                if sample == 0:
                    keys += [(str(row["group"]), "start"), ("all", "start")]
                for dist,subset in keys:
                    counts = stats[name,dist,subset]
                    counts["n"] += 1
                    for attr,t,p in zip(("waypoint_direction","waypoint_distance","gold_direction"),truth,guess,strict=True):
                        counts[attr] += t == p
                    counts["candidate_correct"] += slot_ok
                detailed.append(dict(variant=name,seed=row["scene_seed"],distance=row["group"],sample=sample,
                                     target=truth,prediction=guess,selected=selected[sample],
                                     goal_slot=case["scene"]["goal_slot"],gold_distance=distance[sample]))
    summary = {"/".join(key):dict(value) for key,value in stats.items()}
    return summary,detailed


def run(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if args.device == "cuda" else "cpu"
    if device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one allocated CUDA device")
    args.output.mkdir(exist_ok=False)
    (args.output/"training_config.json").write_text(
        json.dumps({key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
                   indent=2)+"\n")
    plan = json.loads(args.plan.read_text())
    load_weights = partial(torch.load, map_location="cpu", weights_only=True)
    initial = retry_read(load_weights, args.initial)
    first = Predictor(initial,args.condition_candidate).to(device)
    models = dict(no_memory=first, long=deepcopy(first), spatial=deepcopy(first))
    splits = defaultdict(list)
    for row in plan["episodes"]:
        path = args.cache / f"{row['scene_seed']}.pt"
        record = retry_read(partial(load_weights, mmap=True), path)
        if record["row"] != row:
            raise ValueError(f"Cached episode does not match the plan: {path}")
        splits[row["role"]].append(load_case(record, initial["grid_side"], device))
    train = splits["train"]+splits["dart"]
    print("Loaded cached trajectories", {key:len(splits[key]) for key in splits}, flush=True)
    optim = torch.optim.AdamW([p for model in models.values() for p in model.parameters() if p.requires_grad],
                              lr=args.learning_rate, weight_decay=.003)
    with (args.output / "training.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", *VARIANTS))
        for epoch in range(1,args.epochs+1):
            rng.shuffle(train)
            totals = Counter()
            for case in train:
                optim.zero_grad(set_to_none=True)
                losses = {variant:objective(models[variant](case,variant),case) for variant in VARIANTS}
                sum(losses.values()).backward()
                optim.step()
                totals.update({variant:float(loss.detach()) for variant,loss in losses.items()})
            result = [totals[name]/len(train) for name in VARIANTS]
            writer.writerow((epoch,*result))
            stream.flush()
            print(f"epoch {epoch}/{args.epochs} losses {result}",flush=True)
    results = {}
    for label in ("validation","test"):
        results[label],details = evaluate(models,splits[label],interventions=label=="validation")
        (args.output/f"{label}-predictions.json").write_text(json.dumps(details,indent=2)+"\n")
        if label == "validation":
            clean = {(row["seed"],row["sample"]):row["prediction"][:2]
                     for row in details if row["variant"] == "spatial"}
            changed = defaultdict(Counter)
            for row in details:
                if row["variant"] in VARIANTS:
                    continue
                counts = changed[row["variant"]]
                counts["n"] += 1
                counts["changed_waypoint"] += clean[row["seed"],row["sample"]] != row["prediction"][:2]
            results["interventions"] = {name:dict(value) for name,value in changed.items()}
    (args.output/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
    torch.save(dict(initial=args.initial.name,candidate_conditioned=args.condition_candidate,
                    models={k:v.state_dict() for k,v in models.items()}),
               args.output/"waypoint-heads.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan","cache","initial","output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--condition-candidate",action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("Epochs and learning rate must be positive")
    run(args)
