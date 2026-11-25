"""Cache frozen Qwen vision grids and trajectory metadata for lightweight memory tests."""

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor, Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from t1_storage import retry_read
from t1_spatial import pool_vision_grid


DISTANCES = (10, 30, 60, 100)


def select_episodes(dataset, train_per_distance):
    selected = []
    for split in ("train", "validation", "test"):
        groups = defaultdict(list)
        for manifest in sorted((dataset / split).glob("worker-*/episodes.jsonl")):
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                if row["group"] in DISTANCES:
                    groups[row["group"]].append(dict(row, path=str(manifest.parent / row["path"])))
        for distance in DISTANCES:
            rows = groups[distance]
            if split == "train":
                # Spread selected examples across worlds before taking a second one in any world.
                by_world = defaultdict(list)
                for row in rows:
                    by_world[row["world_seed"]].append(row)
                worlds = sorted(by_world, key=lambda world: hashlib.sha256(f"t1-memory:{world}".encode()).digest())
                rows = [by_world[world][slot] for slot in range(max(map(len, by_world.values())))
                        for world in worlds if slot < len(by_world[world])][:train_per_distance]
            if not rows:
                raise ValueError(f"No {distance}-block examples in the {split} manifests")
            selected.extend(rows)
    for key in ("scene_seed", "world_seed"):
        splits = [{row[key] for row in selected if row["split"] == name} for name in ("train", "validation", "test")]
        if any(left & right for i, left in enumerate(splits) for right in splits[i+1:]):
            raise ValueError(f"Cross-split {key} leakage")
    if sum(row["split"] == "validation" for row in selected) != 20:
        raise ValueError("Expected all twenty independent validation trajectories")
    return selected


def load_vision(checkpoint, device):
    config = Qwen2VLConfig.from_pretrained(checkpoint, local_files_only=True)
    config.vision_config._attn_implementation = "sdpa"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    visual = Qwen2VisionTransformerPretrainedModel(config.vision_config)
    state = {}
    names = [name for name in index if name.startswith("visual.")]
    for shard in sorted({index[name] for name in names}):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
            state.update({name.removeprefix("visual."): source.get_tensor(name)
                          for name in names if index[name] == shard})
    visual.load_state_dict(state, strict=True, assign=True)
    return visual.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval(), index


def embedding_norms(checkpoint, index):
    name = "model.embed_tokens.weight"
    with safe_open(checkpoint / index[name], framework="pt", device="cpu") as source:
        table = source.get_slice(name)
        rows, dim = table.get_shape()
        norms = torch.cat([table[i:min(i+2048, rows)].float().norm(dim=-1) for i in range(0, rows, 2048)])
    core = norms[:151643]
    return dict(rows=rows, dimension=dim, all_vocabulary_mean=float(norms.mean()),
                ordinary_vocabulary_rows=len(core), ordinary_mean=float(core.mean()),
                ordinary_median=float(core.median()), ordinary_rms=float(core.square().mean().sqrt()),
                ordinary_p05=float(core.quantile(.05)), ordinary_p95=float(core.quantile(.95)))


def train_indices(meta, stride, return_samples):
    outbound = meta["outbound_frames"]
    count = len(meta["pose"]) - outbound
    if outbound < 2 or count < 2:
        raise ValueError("Expected a nonempty T1 outbound and teacher return")
    history = sorted(set(range(0, outbound, stride)) | {outbound-1})
    frames = sorted({outbound + round(index*(count-1)/(return_samples-1)) for index in range(return_samples)})
    return history, frames


def encode_image_batch(vision, processor, frames, side, device):
    items = processor(images=frames, return_tensors="pt").to(device)
    patches = items["image_grid_thw"]
    values = vision(items["pixel_values"].to(vision.get_dtype()), grid_thw=patches)
    merge = vision.config.spatial_merge_size
    shapes = patches.tolist()
    lengths = [t*h*w//merge**2 for t, h, w in shapes]
    for value, shape in zip(values.split(lengths), shapes, strict=True):
        yield pool_vision_grid(value, shape, merge, side).cpu().half(), value.float().mean(0).cpu().half()


@torch.inference_mode()
def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    vision, weight_index = load_vision(args.checkpoint, args.device)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True).image_processor
    norms = embedding_norms(args.checkpoint, weight_index)
    (args.output / "embedding_norms.json").write_text(json.dumps(norms, indent=2) + "\n")
    rows = select_episodes(args.dataset, args.train_per_distance)
    configuration = dict(vision_path=str(args.checkpoint), resolution=args.resolution, stride=args.stride,
                         spatial_pool_side=args.pool_side, return_samples=args.return_samples, episodes=rows)
    config = args.output / "cache_config.json"
    if config.is_file() and json.loads(config.read_text()) != configuration:
        raise ValueError("Existing cache configuration differs")
    config.write_text(json.dumps(configuration, indent=2) + "\n")
    for number, row in enumerate(rows, 1):
        identifier = row["scene_seed"]
        output = args.output / f"{identifier}.pt"
        if output.exists():
            continue
        archive_path = Path(row["path"])
        local = args.archive_cache / archive_path.name if args.archive_cache else archive_path
        with retry_read(ZipFile, local if local.exists() else archive_path) as archive:
            meta = json.loads(retry_read(archive.read, "episode.json"))
            if (meta["scene"]["scenario_seed"], meta["scene"]["world_seed"], meta["group"], meta["split"]) != (
                    identifier, row["world_seed"], row["group"], row["split"]):
                raise ValueError(f"Archive identity differs: {archive_path}")
            history, current = train_indices(meta, args.stride, args.return_samples)
            indices = history + current
            features, legacy = [], []
            for begin in range(0, len(indices), args.batch):
                frames = []
                for index in indices[begin:begin+args.batch]:
                    data = retry_read(archive.read, f"{index:05d}.jpg")
                    with Image.open(io.BytesIO(data)) as image:
                        frames.append(image.convert("RGB").resize(args.resolution, Image.Resampling.BILINEAR))
                for grid, mean in encode_image_batch(vision, processor, frames, args.pool_side, args.device):
                    features.append(grid)
                    legacy.append(mean)
        temporary = output.with_suffix(".partial")
        torch.save(dict(row=row, meta=meta, history=history, current=current, grid=torch.stack(features),
                        legacy_mean=torch.stack(legacy)), temporary)
        temporary.replace(output)
        print(f"{number}/{len(rows)} {row['split']} d={row['group']} seed={identifier} frames={len(indices)}", flush=True)
    print("feature cache complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--archive-cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--train-per-distance", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--return-samples", type=int, default=12)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--pool-side", type=int, choices=(2, 4), default=2)
    parser.add_argument("--resolution", nargs=2, type=int, default=[320, 180])
    args = parser.parse_args()
    if min(args.train_per_distance, args.stride, args.batch, *args.resolution) < 1 or args.return_samples < 2:
        parser.error("Cache counts, stride, resolution and batch must be positive")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        parser.error("Expose exactly one allocated CUDA device")
    run(args)
