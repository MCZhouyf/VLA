"""Check whether the frozen reader can carry pose when appearance is fixed."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from t1_spatial import SpatialMemory


def rotate_history(poses):
    x, z, _ = poses[-1]
    return [(x - (z_i - z), z + (x_i - x), (yaw + 90) % 360)
            for x_i, z_i, yaw in poses[:-1]] + [poses[-1]]


@torch.inference_mode()
def run(args):
    weights = load_file(args.weights)
    memory = SpatialMemory(weights["write.0.weight"].shape[1], weights["read.1.weight"].shape[0])
    memory.load_state_dict(weights)
    memory.eval()
    with args.index.open(newline="") as stream:
        selected = [row for row in csv.DictReader(stream) if int(row["distance"]) == 30][:5]
    test_rows = []
    for row in selected:
        with gzip.open(Path(row["trace"]), "rt", newline="") as stream:
            poses = [(float(r["x"]), float(r["z"]), float(r["yaw"])) for r in csv.DictReader(stream)]
        rotated = rotate_history(poses)
        features = torch.zeros(len(poses) - 1, memory.write[0].in_features)
        for scale in (0.0, 0.1, 0.5):
            memory.position_value_scale = scale
            original = memory(features, poses)
            changed = memory(features, rotated)
            test_rows.append(dict(seed=int(row["seed"]), history_frames=len(features),
                                  position_value_scale=scale, prefix_norm=float(original.norm()),
                                  rotation_relative_delta=float((original-changed).norm()/original.norm()),
                                  rotation_max_abs_delta=float((original-changed).abs().max())))
    with args.output.open("x") as stream:
        json.dump(dict(appearance="all zero; original learned projection and attention weights",
                       rows=test_rows), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "index", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    run(parser.parse_args())
