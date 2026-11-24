"""Collect matched visible navigation inputs with different target bearings."""

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from t1_collect import NAV_WORLD_SEED, ProtectedCaptureArena, navigation_cases, write_episode
from t1_data_index import read_episodes
from t1_safety import StepTrace
from t1_v4 import DISTANCES, SUCCESS_RADIUS


SCENARIO_BASE = 2073010000
WORLD_BASE = 21170100
DIRECTIONS = ("forward", "left", "right")


def capture_aim(arena, pair_seed, seed, distance, offset, direction):
    config = arena.prepare_visible(pair_seed, distance, offset)
    config.update(scenario_seed=seed, steering_pair_seed=pair_seed, steering_target=direction)
    arena.capture = []
    try:
        with arena.track(StepTrace(None)):
            for _ in range(2):
                arena.teacher_step()
            outbound = arena.capture
            arena.capture = []
            steps, _, _ = arena.drive(arena.landmark, SUCCESS_RADIUS, 1)
        returned = arena.capture
        if steps != 1 or len(returned) != 1:
            raise RuntimeError("The visible start did not produce one teacher action")
        return config, outbound, returned
    finally:
        arena.capture = None


def collect(args):
    world = WORLD_BASE + args.worker + args.world_offset
    arena = ProtectedCaptureArena(world)
    try:
        with (args.output / "episodes.jsonl").open("x") as index:
            for position in range(args.positions_per_worker):
                pair_seed = SCENARIO_BASE + 3 * (100 * args.worker + position)
                rng = random.Random(f"paired-steering:{pair_seed}")
                distance = rng.uniform(5.5, 9.5)
                angle = rng.uniform(17.0, 32.0)
                # Positive camera yaw turns right, so a positive start offset needs a left correction.
                offsets = dict(forward=0.0, left=angle, right=-angle)
                order = list(enumerate(DIRECTIONS))
                rng.shuffle(order)
                triplet = [capture_aim(arena, pair_seed, pair_seed + i, distance, offsets[direction], direction)
                           for i, direction in order]
                for config, outbound, returned in triplet:
                    row = write_episode(args.output, config, outbound, returned, "train", "visible")
                    index.write(json.dumps(row) + "\n")
                    index.flush()
                    print(json.dumps(row), flush=True)
    finally:
        arena.sim.close()


def freeze(args):
    records = read_episodes(args.roots)
    frozen = json.loads(args.frozen.read_text())
    benchmark = set(frozen["pilot"]) | set(frozen["formal"])
    scenes = benchmark | {case[0] for offset in (False, True) for case in navigation_cases(offset)}
    worlds = {frozen["world_seed_by_scenario"][str(seed)] for seed in benchmark} | {NAV_WORLD_SEED}
    for source in args.existing:
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        scenes.update(row["scene_seed"] for row in rows)
        worlds.update(row["world_seed"] for row in rows)
    if {row["scene_seed"] for row in records} & scenes or {row["world_seed"] for row in records} & worlds:
        raise ValueError("Paired scenes or worlds overlap a prior split or frozen benchmark")
    paired = defaultdict(list)
    index = []
    for row in records:
        with ZipFile(row["path"]) as archive:
            meta = json.loads(archive.read("episode.json"))
        config = meta["scene"]
        target = config["steering_target"]
        if row["split"] != "train" or row["group"] != "visible" or row["frames"] != 3:
            raise ValueError(f"Invalid paired observation layout for {row['scene_seed']}")
        if row["outbound_frames"] != 2 or row["return_frames"] != 1:
            raise ValueError(f"Expected two observations and one action for {row['scene_seed']}")
        ids = [int(token) for token in re.findall(r"reserved_special_token_(\d+)", meta["action"][-1])]
        expected = {"forward": (240, True), "left": (230, False), "right": (250, False)}
        if (ids[-2], 191 in ids) != expected[target]:
            raise ValueError(f"Expected {target} teacher action for {row['scene_seed']}")
        pair = config["steering_pair_seed"]
        paired[pair].append((row["world_seed"], target, meta["pose"][-1][:2]))
        index.append(dict(row, steering_pair_seed=pair, steering_target=target,
                          selected_frames=3, selected_labels=1))
    if len(paired) != args.expected_positions:
        raise ValueError(f"Found {len(paired)} paired positions, expected {args.expected_positions}")
    for pair, poses in paired.items():
        if (sorted(t for _, t, _ in poses) != sorted(DIRECTIONS)
                or len({world for world, _, _ in poses}) != 1
                or max(math.dist(a[2], b[2]) for a in poses for b in poses) > 0.02):
            raise ValueError(f"Teacher triplet {pair} did not share a start position and world")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in index))
    summary = dict(paired_positions=len(paired), world_count=len({row["world_seed"] for row in records}),
                   frames=3 * len(index), supervised_actions=len(index),
                   examples_per_direction=dict.fromkeys(DIRECTIONS, len(paired)),
                   scene_overlap=0, world_overlap=0, index_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def replay_index(args):
    import cv2
    import numpy as np

    from t1_shortcut_audit import gold_pixels

    paired = [dict(json.loads(line), control_training_source="paired")
              for line in args.paired_index.read_text().splitlines()]
    original = [json.loads(line) for line in args.base_index.read_text().splitlines()]
    if not paired or len(paired) % 6:
        raise ValueError("A replay index requires an even number of complete steering triplets")
    horizontal_gold = defaultdict(list)
    pillar_pixels = []
    for row in paired:
        with ZipFile(row["path"]) as archive:
            image = cv2.imdecode(np.frombuffer(archive.read("00002.jpg"), np.uint8), cv2.IMREAD_COLOR)
            meta = json.loads(archive.read("episode.json"))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x, z, yaw = meta["pose"][-1]
        scene = meta["scene"]
        y = scene["return_pose"]["y"]
        gold_x, gold_z = scene["landmark"]
        pillar = (math.floor(gold_x), math.floor(y), math.floor(gold_z))
        pixels = gold_pixels(rgb, None, (x, y, z, yaw, 0.0), pillar)
        angle = math.radians(yaw)
        dx, dz = gold_x - x, gold_z - z
        depth = dz * math.cos(angle) - dx * math.sin(angle)
        if depth <= 0:
            raise ValueError(f"Gold is behind paired camera for {row['scene_seed']}")
        lateral = -dx * math.cos(angle) - dz * math.sin(angle)
        focal = rgb.shape[0] / (2 * math.tan(math.radians(35)))
        center = 0.5 + focal * lateral / (rgb.shape[1] * depth)
        target = row["steering_target"]
        if center < 0.45:
            visible_side = "left"
        elif center > 0.55:
            visible_side = "right"
        else:
            visible_side = "forward"
        if not pixels or not 0 < center < 1 or visible_side != target:
            raise ValueError(f"Projected gold pixels disagree with steering label for {row['scene_seed']}")
        horizontal_gold[target].append(center)
        pillar_pixels.append(pixels)
    groups = (*DISTANCES, "visible")
    per_source = len(paired) // 2
    split, remainder = divmod(per_source, len(groups))
    selected = []
    for source in ("original", "dart"):
        for i, group in enumerate(groups):
            remaining = split + int(i < remainder)
            candidates = [row for row in original if row["split"] == "train"
                          and row.get("control_training_source") == source and row["group"] == group]
            random.Random(f"paired-replay:{source}:{group}").shuffle(candidates)
            for row in candidates:
                if not remaining:
                    break
                use = min(remaining, row["selected_labels"])
                if use:
                    selected.append(dict(row, selected_frames=row["outbound_frames"] + use,
                                         selected_labels=use))
                    remaining -= use
            if remaining:
                raise ValueError(f"Insufficient replay labels in {source}/{group}")
    rows = paired + selected + [row for row in original if row["split"] != "train"]
    if len({r["scene_seed"] for r in rows}) != len(rows):
        raise ValueError("Steering pairs and replay may not reuse any scene seeds")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    index = output / "dataset.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = dict(
        paired_positions=len(paired) // 3, paired_labels=len(paired), replay_labels_by_source=
        dict.fromkeys(("original", "dart"), per_source), labels_per_epoch=len(paired) + 2 * per_source,
        original_index_sha256=hashlib.sha256(args.base_index.read_bytes()).hexdigest(),
        paired_index_sha256=hashlib.sha256(args.paired_index.read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        gold_pixels_side_verified=len(paired),
        minimum_projected_gold_pixels=min(pillar_pixels),
        gold_horizontal_fraction_ranges={direction: [min(xs), max(xs)]
                                         for direction, xs in horizontal_gold.items()},
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture = modes.add_parser("collect")
    capture.add_argument("output", type=Path)
    capture.add_argument("--worker", type=int, required=True)
    capture.add_argument("--workers", type=int, required=True)
    capture.add_argument("--positions-per-worker", type=int, default=4)
    capture.add_argument("--world-offset", type=int, default=0)
    capture.set_defaults(run=collect)
    audit = modes.add_parser("freeze")
    audit.add_argument("output", type=Path)
    audit.add_argument("roots", nargs="+", type=Path)
    audit.add_argument("--frozen", type=Path, required=True)
    audit.add_argument("--existing", type=Path, nargs="+", required=True)
    audit.add_argument("--expected-positions", type=int, required=True)
    audit.set_defaults(run=freeze)
    replay = modes.add_parser("replay-index")
    replay.add_argument("output", type=Path)
    replay.add_argument("--paired-index", type=Path, required=True)
    replay.add_argument("--base-index", type=Path, required=True)
    replay.set_defaults(run=replay_index)
    args = parser.parse_args()
    if args.mode == "collect" and (not 0 <= args.worker < args.workers <= 32
                                    or not 0 < args.positions_per_worker <= 100
                                    or args.world_offset < 0):
        parser.error("Invalid steering collection shard or position count")
    args.run(args)


if __name__ == "__main__":
    main()
