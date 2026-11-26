"""Cache a common set of T1 v3 observations for the three waypoint predictors."""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


DISTANCES = (10, 30, 60, 100)
RESOLUTION = (320, 180)
OUTBOUND_STRIDE = 8
HISTORY_LIMIT = 8
RETURN_SAMPLES = 24
VISIBLE_PER_TRAIN_ROLE = 16


def read_rows(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def verify_gate(source, fixtures):
    scores = json.loads(source.read_text())
    gated = scores["distances"]
    if not scores["passed"] or any(gated[str(d)]["n"] != 30 or gated[str(d)]["successes"] < 24
                                   for d in DISTANCES):
        raise ValueError("The recovered four-distance oracle gate has not passed")
    return {case["world_seed"] for case in fixtures["fixtures"].values()}, set(fixtures["formal"])


def held_out(run, role):
    rows = []
    for manifest in sorted((run / "data" / role).glob("worker-*/episodes.jsonl")):
        for row in read_rows(manifest):
            if row["group"] in DISTANCES:
                rows.append(dict(row, role=role, path=str(manifest.parent / row["path"]),
                                 selected_labels=row["return_frames"]))
    return rows


def make_plan(run, oracle_gate, output):
    index = run / "index/dataset.jsonl"
    checksum = hashlib.sha256(index.read_bytes()).hexdigest()
    if checksum != json.loads((run / "index/summary.json").read_text())["index_sha256"]:
        raise ValueError("Frozen waypoint index checksum does not match")
    fixtures = json.loads((run / "frozen/fixtures.json").read_text())
    frozen_worlds, frozen_seeds = verify_gate(oracle_gate, fixtures)
    training = [row for row in read_rows(index) if row["role"] in ("train", "dart") and row["selected_labels"]]
    visible_count = Counter()
    rows = []
    for row in training:
        role = row["role"]
        if row["group"] == "visible":
            if visible_count[role] >= VISIBLE_PER_TRAIN_ROLE:
                continue
            visible_count[role] += 1
        rows.append(row)
    rows += held_out(run, "validation") + held_out(run, "test")
    prior_seeds = set(frozen_seeds)
    prior_worlds = set(frozen_worlds)
    for role in ("train", "dart", "validation", "test"):
        group = [row for row in rows if row["role"] == role]
        scenes = {row["scene_seed"] for row in group}
        worlds = {row["world_seed"] for row in group}
        if (len(scenes) != len(group) or scenes & prior_seeds or worlds & prior_worlds
                or not set(DISTANCES).issubset(row["group"] for row in group)):
            raise ValueError(f"Bad {role} coverage or seed overlap")
        prior_seeds.update(scenes)
        prior_worlds.update(worlds)
    config = dict(index_sha256=checksum, oracle_gate=str(oracle_gate),
                  archive_cache=(run / "archive_cache").read_text().strip(),
                  resolution=RESOLUTION, patch_grid=2, outbound_stride=OUTBOUND_STRIDE,
                  history_limit=HISTORY_LIMIT, return_samples=RETURN_SAMPLES, episodes=rows)
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(dict(Counter(f"{row['role']}/{row['group']}" for row in rows)), flush=True)


def frame_indices(outbound, labels):
    if outbound < 1 or labels < 1:
        raise ValueError("Expected an outbound history and return supervision")
    history = sorted(set(range(0, outbound, OUTBOUND_STRIDE)) | {outbound-1})
    targets = sorted({outbound + round(i*(labels-1)/(RETURN_SAMPLES-1)) for i in range(RETURN_SAMPLES)})
    windows = {frame for i in targets for frame in range(max(0, i-HISTORY_LIMIT), i+1)}
    return history, targets, sorted(set(history) | windows)


def encode(plan, checkpoint, output, shard, batch):
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    from t1_memory_cache import encode_image_batch, load_vision
    from t1_storage import retry_read

    if torch.cuda.device_count() != 1:
        raise ValueError("Expose one allocated CUDA device")
    config = json.loads(plan.read_text())
    rows = config["episodes"][shard::2]
    vision, _ = load_vision(checkpoint, "cuda")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True).image_processor
    output.mkdir(exist_ok=True)
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            seed = row["scene_seed"]
            dst = output / f"{seed}.pt"
            if dst.is_file():
                continue
            source = Path(row["path"])
            cached = Path(config["archive_cache"]) / source.name
            with retry_read(ZipFile, cached if cached.is_file() else source) as archive:
                meta = json.loads(retry_read(archive.read, "episode.json"))
                pose = meta["scene"]
                if (pose["scenario_seed"], pose["world_seed"], meta["group"]) != (seed, row["world_seed"], row["group"]):
                    raise ValueError(f"Archive identity differs for {seed}")
                history, labels, frames = frame_indices(meta["outbound_frames"], row["selected_labels"])
                if frames[-1] >= len(meta["pose"]):
                    raise ValueError(f"Invalid frame indices for {seed}")
                features = []
                for offset in range(0, len(frames), batch):
                    images = []
                    for frame in frames[offset:offset+batch]:
                        pixels = retry_read(archive.read, f"{frame:05d}.jpg")
                        with Image.open(io.BytesIO(pixels)) as image:
                            images.append(image.convert("RGB").resize(RESOLUTION, Image.Resampling.BILINEAR))
                    features.extend(grid for grid, _ in encode_image_batch(vision, processor, images, 2, "cuda"))
            temporary = dst.with_suffix(".partial")
            torch.save(dict(row=row, meta=meta, frame_indices=frames, history=history, labels=labels,
                            grid=torch.stack(features)), temporary)
            temporary.replace(dst)
            print(f"{i}/{len(rows)} {row['role']} d={row['group']} seed={seed} features={len(frames)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("plan")
    for name in ("run", "oracle-gate", "output"):
        build.add_argument(f"--{name}", required=True, type=Path)
    cache = commands.add_parser("encode")
    for name in ("plan", "checkpoint", "output"):
        cache.add_argument(f"--{name}", required=True, type=Path)
    cache.add_argument("--shard", required=True, type=int, choices=(0, 1))
    cache.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args.run, args.oracle_gate, args.output)
    else:
        if args.batch < 1:
            cache.error("Batch must be positive")
        encode(args.plan, args.checkpoint, args.output, args.shard, args.batch)
