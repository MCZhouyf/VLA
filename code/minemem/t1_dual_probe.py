"""Train three waypoint selectors on matched dual-enclosure observations."""

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor

from t1_memory_cache import encode_image_batch, load_vision
from t1_spatial import GeometricSpatialMemory, spatial_read_plan
from t1_storage import retry_read


VARIANTS = ("no_memory", "long", "spatial")
SPLITS = ("train", "validation", "test")
DISTANCES = (10, 30, 60, 100)


def read_collection(collection):
    selected = []
    for split in SPLITS:
        for report in sorted((collection / split).glob("worker-*/oracle.csv")):
            with report.open(newline="") as stream:
                records = list(csv.DictReader(stream))
            for row in records:
                if row["error"]:
                    raise ValueError(f"Incomplete {split} scene in {report}: {row['error']}")
                seed, distance, slot = (int(row[key]) for key in ("seed", "distance", "goal_slot"))
                path = report.parent / f"d{distance}-seed{seed}-goal{slot}-observations.zip"
                selected.append(dict(split=split, seed=seed, distance=distance, slot=slot,
                                     world=int(row["world_seed"]), oracle=row["success"] == "True",
                                     identical_headings=int(row["identical_headings"]), path=str(path)))
    for split in SPLITS:
        for distance in DISTANCES:
            cases = [row for row in selected if row["split"] == split and row["distance"] == distance]
            if not cases or sum(row["oracle"] for row in cases)/len(cases) < .80:
                raise ValueError(f"True waypoint controller gate failed for {split}/{distance}")
    pairs = defaultdict(list)
    for row in selected:
        pairs[row["seed"]].append(row)
    if any(len(pair) != 2 or {row["slot"] for row in pair} != {0, 1}
           or len({row["split"] for row in pair}) != 1 for pair in pairs.values()):
        raise ValueError("Collection does not have both goal assignments for every paired seed")
    for seed, pair in pairs.items():
        observed = []
        for row in pair:
            with retry_read(ZipFile, row["path"]) as archive:
                scene = json.loads(retry_read(archive.read, "episode.json"))["scene"]
                if scene["candidate_occlusion"] != [True, True]:
                    raise ValueError(f"A candidate is not occluded at the endpoint for {seed}")
                first = scene["history_frames"]
                digests = [hashlib.sha256(retry_read(archive.read, f"{index:05d}.jpg")).hexdigest()
                           for index in range(first, first+8)]
                if digests != scene["shared_endpoint_window_sha256"]:
                    raise ValueError(f"Endpoint window metadata differs for {seed}")
                observed.append(digests)
        if observed[0] != observed[1]:
            raise ValueError(f"The two goal assignments do not share an endpoint window for {seed}")
    worlds = [{row["world"] for row in selected if row["split"] == part} for part in SPLITS]
    if any(left & right for i,left in enumerate(worlds) for right in worlds[i+1:]):
        raise ValueError("World seed leakage between splits")
    return selected


@torch.inference_mode()
def cache_features(args):
    rows = read_collection(args.collection)
    args.output.mkdir(parents=True, exist_ok=True)
    vision, _ = load_vision(args.vision, args.device)
    processor = AutoProcessor.from_pretrained(args.vision, local_files_only=True).image_processor
    for n, row in enumerate(rows, 1):
        target = args.output / f"{row['seed']}-{row['slot']}.pt"
        if target.exists():
            continue
        with retry_read(ZipFile, row["path"]) as archive:
            meta = json.loads(retry_read(archive.read, "episode.json"))
            scene = meta["scene"]
            if (scene["scenario_seed"], scene["goal_slot"], meta["split"]) != (row["seed"], row["slot"], row["split"]):
                raise ValueError(f"Scene metadata differs from the collection for {row['path']}")
            poses = meta["pose"]
            if len(poses) - scene["history_frames"] != 8:
                raise ValueError("The current observation window must have eight simulator frames")
            features = []
            for begin in range(0, len(poses), 6):
                frames = []
                for index in range(begin, min(len(poses), begin+6)):
                    with Image.open(io.BytesIO(retry_read(archive.read, f"{index:05d}.jpg"))) as image:
                        frames.append(image.convert("RGB").resize((320, 180), Image.Resampling.BILINEAR))
                features.extend(grid for grid, _ in encode_image_batch(vision, processor, frames, 2, args.device))
        temporary = target.with_suffix(".partial")
        torch.save(dict(row=row, metadata=meta, features=torch.stack(features)), temporary)
        temporary.replace(target)
        print(f"{n}/{len(rows)} {row['split']} d={row['distance']} seed={row['seed']} goal={row['slot']}", flush=True)
    (args.output / "index.json").write_text(json.dumps(rows, indent=2)+"\n")


class WindowSelector(nn.Module):
    def __init__(self, dimension, window):
        super().__init__()
        self.window = window
        self.project = nn.Sequential(nn.Linear(dimension, 16, bias=False), nn.LayerNorm(16))
        self.head = nn.Linear(window*64, 2)

    def forward(self, case):
        tokens = self.project(case["current"][-self.window:]).flatten()
        return self.head(tokens)


class SpatialSelector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.memory = model
        self.memory.to_llm.requires_grad_(False)
        self.head = nn.Linear(model.width, 1)

    def forward(self, case, shuffle=False):
        indices, geometry, ranks = case["plan"]
        history = case["history"]
        if shuffle:
            history = history[case["permutation"]]
        values = self.memory.read(history[indices], geometry, ranks)
        return self.head(values[:2]).flatten()


def build_case(episode, mean, std, side):
    metadata = episode["metadata"]
    scene = metadata["scene"]
    poses = metadata["pose"]
    n = scene["history_frames"]
    plan = spatial_read_plan(poses[:n], poses[n], side, scene["room_frame_indices"])
    value = ((episode["features"].float()-mean)/std).clamp(-8, 8)
    rng = np.random.default_rng(episode["row"]["seed"])
    return dict(row=episode["row"], history=value[:n], current=value[n:],
                plan=tuple(torch.from_numpy(part) for part in plan),
                permutation=torch.from_numpy(rng.permutation(n)))


@torch.no_grad()
def score(models, examples):
    counts = Counter()
    choices = {}
    for case in examples:
        row = case["row"]
        outcome = {variant: int(model(case).argmax()) for variant, model in models.items()}
        outcome["shuffled_visual"] = int(models["spatial"](case, shuffle=True).argmax())
        choices[f"{row['seed']}:{row['slot']}"] = outcome
        for metric in (row["distance"], "all"):
            counts[metric, "episodes"] += 1
            for variant, selected in outcome.items():
                counts[metric, variant] += int(selected == row["slot"])
    summary = {metric: {key: counts[metric, key] for key in ("episodes", *VARIANTS, "shuffled_visual")}
               for metric in (*DISTANCES, "all")}
    return summary, choices


def train(args):
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    initial = torch.load(args.initial, map_location="cpu", weights_only=True)
    if initial["patch_side"] != 2:
        raise ValueError("This paired pilot caches 2×2 vision grids")
    mem = GeometricSpatialMemory(initial["vision_dim"], initial["embedding_dim"], initial["embedding_norm"],
                                 initial["patch_side"], grid_side=initial["grid_side"])
    mem.load_state_dict({k.removeprefix("memory."): v for k,v in initial["model"].items()
                        if k.startswith("memory.")}, strict=True)
    models = dict(no_memory=WindowSelector(initial["vision_dim"], 1),
                  long=WindowSelector(initial["vision_dim"], 8), spatial=SpatialSelector(mem))
    for variant in ("no_memory", "long"):
        models[variant].project[0].load_state_dict(mem.patch.state_dict())
    rows = json.loads((args.cache / "index.json").read_text())
    examples = {split: [] for split in SPLITS}
    for row in rows:
        record = torch.load(args.cache / f"{row['seed']}-{row['slot']}.pt", map_location="cpu", weights_only=True)
        case = build_case(record, initial["mean"], initial["std"], initial["grid_side"])
        examples[row["split"]].append(case)
    parameters = [parameter for model in models.values() for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.0005, weight_decay=.003)
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs+1):
        rng.shuffle(examples["train"])
        for begin in range(0, len(examples["train"]), 8):
            batch = examples["train"][begin:begin+8]
            labels = torch.tensor([case["row"]["slot"] for case in batch])
            loss = sum(F.cross_entropy(torch.stack([model(case) for case in batch]), labels)
                       for model in models.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch} final_batch_loss={float(loss.detach()):.4f}", flush=True)
    summary = {}
    for split in SPLITS:
        summary[split], predictions = score(models, examples[split])
        (args.output / f"{split}-selections.json").write_text(json.dumps(predictions, indent=2)+"\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    torch.save({key: model.state_dict() for key,model in models.items()}, args.output / "waypoint-selectors.pt")
    print(json.dumps({split: results["all"] for split,results in summary.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    cache = subcommands.add_parser("cache")
    for name in ("collection", "vision", "output"):
        cache.add_argument(f"--{name}", type=Path, required=True)
    cache.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    fit = subcommands.add_parser("train")
    for name in ("cache", "initial", "output"):
        fit.add_argument(f"--{name}", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    if args.command == "cache":
        if args.device == "cuda" and torch.cuda.device_count() != 1:
            parser.error("Expose only one allocated device")
        cache_features(args)
    else:
        if args.epochs < 1:
            parser.error("Training epochs must be positive")
        train(args)
