"""Fit and score a room readout using only uniformly sampled outbound frames."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import nn
from torch.nn import functional as F

from t1_phase3e_pair_probe import RoomReadout, TAIL, load_cache, summarise, train_rows
from t1_storage import retry_read


STRIDE = 8


class AnchorFreeReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(width + 4, 64), nn.LayerNorm(64), nn.GELU())
        self.attention = nn.Linear(64, 1)
        self.room = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, visual, poses):
        # Define slot order from the observed outbound axis. The return yaw is immaterial.
        end = poses[-1, :2] - poses[0, :2]
        forward = F.normalize(end, dim=0)
        sideways = torch.stack((forward[1], -forward[0]))
        delta = poses[:, :2] - poses[0, :2]
        lateral = delta @ sideways
        heading = torch.deg2rad(poses[:, 2])
        look = torch.stack((-heading.sin(), heading.cos()), dim=-1)
        geometry = torch.stack((lateral / 16, delta @ forward / 64,
                                look @ sideways, look @ forward), dim=-1)
        values = self.encode(torch.cat((visual, geometry), dim=-1))
        scores = self.attention(values).squeeze(-1)
        sides = torch.stack((-lateral, lateral))
        weights = (scores + F.logsigmoid(sides / 2)).softmax(dim=-1)
        return self.room(weights @ values).squeeze(-1)


def metadata(path):
    with retry_read(ZipFile, path) as archive:
        return json.loads(retry_read(archive.read, "episode.json"))


def uniform_indices(history, start):
    if not history or sorted(set(history)) != history or history[-1] >= start - TAIL:
        raise ValueError("Invalid ordering in cached outbound frames")
    positions = {frame: index for index, frame in enumerate(history)}
    tail = start - TAIL
    indices = [positions[frame] if frame < tail else len(history) + frame - tail
               for frame in range(0, start, STRIDE)]
    return torch.tensor(indices), len(history)


@torch.inference_mode()
def project_record(record, history, start, tail_poses, encoder):
    indices, size = uniform_indices(history, start)
    if size != len(record["history_pose"]) or len(tail_poses) != TAIL:
        raise ValueError(f"Outward cache/metadata mismatch at seed {record['seed']}")
    poses = torch.tensor([*record["history_pose"], *tail_poses])[indices]
    return [(int(slot), encoder.embed(torch.cat((data["history"], record["shared"][:TAIL]))[indices]), poses)
            for slot, data in record["arms"].items()]


def training_cases(plan, cache, encoder):
    archive_cache = Path(json.loads(plan.read_text())["archive_cache"])
    cases = []
    for row in train_rows(plan):
        record = load_cache(cache, row["scene_seed"])
        if (record["seed"], record["split"], record["world"]) != (row["scene_seed"], "train", row["world_seed"]):
            raise ValueError("Training cache identity changed")
        original = Path(row["path"])
        local = archive_cache / original.name
        meta = metadata(local if local.is_file() else original)
        start = meta["outbound_frames"]
        # Anchor offsets reconstruct the old cache ordering only. Non-stride frames never reach the model.
        history = sorted(set(range(0, start - TAIL, STRIDE)) | set(meta["scene"]["room_frame_indices"]))
        case, = project_record(record, history, start, meta["pose"][start-TAIL:start], encoder)
        cases.append((record["seed"], record["world"], *case))
    return cases


def paired_cases(holdout, cache, encoder):
    for row in json.loads(holdout.read_text())["archives"]:
        record = load_cache(cache, row["seed"])
        if any(record[key] != row[key] for key in ("split", "seed", "world", "distance")):
            raise ValueError("Held-out cache identity changed")
        meta = metadata(row["path"])
        history = meta["history_indices"]
        frames = meta["rendered_poses"][str(meta["slots"][0])]
        tail_poses = [[p[0], p[2], p[3]] for p in frames[len(history):len(history)+TAIL]]
        start = len(meta["canonical_outbound_pose"]) - 1
        yield row, project_record(record, history, start, tail_poses, encoder)


def run(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    encoder = RoomReadout(checkpoint["specification"])
    encoder.load_state_dict(checkpoint["models"]["spatial"])
    encoder.eval().requires_grad_(False)
    readout = AnchorFreeReadout(encoder.reader.width - 32)
    cases = training_cases(args.train_plan, args.cache, encoder)
    # Projection is frozen. Clone once outside inference_mode for autograd in the new attention head.
    cases = [(seed, world, slot, visual.clone(), pose.clone()) for seed, world, slot, visual, pose in cases]
    worlds = {world for _, world, *_ in cases}
    holdouts = list(paired_cases(args.holdout, args.cache, encoder))
    if worlds & {row["world"] for row, _ in holdouts}:
        raise ValueError("Training and held-out world seeds overlap")
    args.output.mkdir()
    optim = torch.optim.AdamW(readout.parameters(), lr=.001)
    with (args.output / "training.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "loss", "train_accuracy"))
        for epoch in range(1, args.epochs + 1):
            rng.shuffle(cases)
            total = wins = 0
            for _, _, slot, visual, pose in cases:
                score = readout(visual, pose)
                loss = F.cross_entropy(score[None], torch.tensor([slot]))
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                total += loss.item()
                wins += int(score.argmax() == slot)
            writer.writerow((epoch, total / len(cases), wins / len(cases)))
            stream.flush()
            print(f"epoch={epoch}/{args.epochs} loss={total/len(cases):.5f} train={wins}/{len(cases)}", flush=True)
    readout.eval()
    predictions = []
    with torch.inference_mode():
        for row, arms in holdouts:
            for slot, visual, poses in arms:
                permutation = torch.randperm(len(poses), generator=torch.Generator().manual_seed(row["seed"]))
                for variant, values in (("no_anchor", visual), ("shuffle_visual", visual[permutation])):
                    choice = readout(values, poses).argmax().item()
                    # Only the room decision is computed. No waypoint score is included.
                    predictions.append(dict(split=row["split"], seed=row["seed"], world=row["world"],
                                            distance=row["distance"], slot=slot, primary=slot == int(row["gold"]),
                                            state=0, variant=variant, room=choice))
    metrics = summarise(predictions, include_direction=False)
    passed = {d: all(metrics[f"test/{d}/0/no_anchor"][key]["rate"] >= limit
                     for key, limit in (("room", .85), ("flip", .80))) for d in (10, 30, 60, 100, "all")}
    weights = args.output / "room-head.pt"
    torch.save(dict(specification=checkpoint["specification"], room_state=checkpoint["models"]["spatial"],
                    attention_state=readout.state_dict()), weights)
    summary = dict(primary_distance=30, passed=passed[30], passed_by_distance=passed, stride=STRIDE,
                   train_seeds=sorted(seed for seed, *_ in cases), epochs=args.epochs,
                   frozen_encoder_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                   weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), metrics=metrics)
    (args.output / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    (args.output / "offline_gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(passed), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-plan", "cache", "weights", "holdout", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261126)
    run(parser.parse_args())
