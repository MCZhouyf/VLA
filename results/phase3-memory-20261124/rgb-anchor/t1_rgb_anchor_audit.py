"""Test whether outbound RGB keyframes can identify a useful goal anchor."""

import argparse
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from t1_shortcut_audit import gold_target
from t1_storage import retry_read
from t1_v4 import SUCCESS_RADIUS


def evaluate(archive, metadata, stride):
    outbound = metadata["outbound_frames"]
    dwell = metadata["scene"]["outbound_dwell_steps"]
    if not 0 < dwell < outbound:
        raise ValueError("Expected a dwell and a subsequent outbound route")
    indices = sorted(set(range(0, outbound, stride)) | set(range(dwell, outbound, stride)))
    observations = {}
    for index in indices:
        data = retry_read(archive.read, f"{index:05d}.jpg")
        with Image.open(io.BytesIO(data)) as image:
            frame = np.asarray(image.convert("RGB"))
        observations[index] = gold_target(frame)[0]

    def best(candidates):
        candidate = max(candidates, key=observations.__getitem__)
        return candidate if observations[candidate] else None

    actual = best(indices)
    after_dwell = best([index for index in indices if index >= dwell])
    shuffled = indices.copy()
    seed = metadata["scene"]["scenario_seed"]
    random.Random(f"rgb-anchor:{seed}").shuffle(shuffled)
    association = dict(zip(indices, shuffled, strict=True))
    picks = {
        "first_pose": (0, 0),
        "rgb_anchor": (actual, actual),
        "rgb_anchor_without_dwell": (after_dwell, after_dwell),
        "rgb_at_shuffled_pose": (actual, association.get(actual)),
    }
    landmark = metadata["scene"]["landmark"]
    for method, (observation, pose) in picks.items():
        distance = None if pose is None else math.dist(metadata["pose"][pose][:2], landmark)
        yield dict(seed=seed, world=metadata["scene"]["world_seed"], split=metadata["split"],
                   d=metadata["group"], method=method, chosen_observation=observation, chosen_pose=pose,
                   candidate_frames=len(indices), gold_pixels=observations.get(observation),
                   goal_distance=distance, within_success_radius=distance is not None and distance <= SUCCESS_RADIUS)


def run(args):
    index = retry_read(args.dataset.read_bytes)
    held_out = [row for line in index.splitlines() if (row := json.loads(line))["split"] in {"validation", "test"}
                and row["group"] in {30, 60, 100}]
    if not held_out:
        raise ValueError("The index has no distant held-out episodes")
    output = []
    for row in held_out:
        path = args.archive_cache / Path(row["path"]).name
        with retry_read(ZipFile, path) as archive:
            metadata = json.loads(retry_read(archive.read, "episode.json"))
            if (metadata["scene"]["scenario_seed"] != row["scene_seed"]
                    or metadata["group"] != row["group"] or metadata["split"] != row["split"]):
                raise ValueError(f"Archive lineage differs from the index for {path}")
            output.extend(evaluate(archive, metadata, args.stride))
    args.output.mkdir(exist_ok=False)
    with (args.output / "anchors.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {}
    for distance in (30, 60, 100):
        summary[distance] = {}
        for method in sorted({row["method"] for row in output}):
            cases = [row for row in output if row["d"] == distance and row["method"] == method]
            summary[distance][method] = dict(episodes=len(cases), within_3=sum(row["within_success_radius"]
                                                                            for row in cases))
    with (args.output / "summary.json").open("x") as stream:
        json.dump(dict(dataset_sha256=hashlib.sha256(index).hexdigest(), stride=args.stride,
                       result=summary), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "archive-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("History stride must be positive")
    run(args)
