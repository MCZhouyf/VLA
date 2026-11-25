"""Freeze accepted T1 v3 scenes and the separate teacher/DART data splits."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from phase1_metrics import wilson
from t1_collect import DART_QUOTAS, HELD_OUT_QUOTAS, TRAIN_QUOTAS
from t1_data_index import read_episodes, select
from t1_phase3e import ACCEPTANCE_SEEDS, BASELINES, DISTANCES, GOLD_ORDER, PROTOCOL


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+"\n")


def incomplete_shards(source, split, workers):
    quotas = {"train": TRAIN_QUOTAS, "dart": DART_QUOTAS}.get(split, HELD_OUT_QUOTAS)
    missing = []
    for index in range(workers):
        worker = source / f"worker-{index:03d}"
        counts = Counter()
        try:
            for line in (worker / "episodes.jsonl").read_text().splitlines():
                row = json.loads(line)
                archive = worker / row["path"]
                if archive.parent != worker or not archive.is_file():
                    raise ValueError(f"Missing archive: {archive}")
                counts[row["group"]] += row["frames"]
        except (OSError, ValueError, KeyError):
            counts.clear()
        if any(counts[group] < math.ceil(quota/workers) for group, quota in quotas.items()):
            missing.append(index)
    return missing


def acceptance(source, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = {}
    configs = {}
    gold_slots = dict(zip(ACCEPTANCE_SEEDS, GOLD_ORDER, strict=True))
    for result in sorted(source.glob("worker-*/acceptance.csv")):
        with result.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["error"]:
                    raise ValueError(f"Unresolved setup failure in {result}: {row['error']}")
                d, seed = int(row["distance"]), int(row["seed"])
                if int(row["goal_slot"]) != gold_slots[seed]:
                    raise ValueError(f"Assignment differs from the balanced frozen seed {seed}: {result}")
                key = (d, seed, row["baseline"])
                if key in rows:
                    raise ValueError(f"Duplicate evaluation result: {key}")
                rows[key] = row
                if row["baseline"] == "teacher":
                    config = result.parent / f"d{d}-seed{seed}.json"
                    configs[f"{d}/{seed}"] = json.loads(config.read_text())
    if set(rows) != {(d, seed, baseline) for d in DISTANCES for seed in ACCEPTANCE_SEEDS for baseline in BASELINES}:
        raise ValueError("Incomplete teacher and shortcut coverage on the frozen scenes")
    metrics = {}
    for d in DISTANCES:
        metrics[d] = {}
        for baseline in BASELINES:
            cases = [rows[d, seed, baseline] for seed in ACCEPTANCE_SEEDS]
            successes = sum(row["success"] == "True" for row in cases)
            metrics[d][baseline] = dict(n=len(cases), successes=successes, rate=successes/len(cases),
                                        deaths=sum(int(row["deaths"]) for row in cases),
                                        pose_jumps=sum(int(row["pose_jumps"]) for row in cases))
        reverse = metrics[d]["reverse_route"]
        low, high = wilson(reverse["successes"], reverse["n"])
        reverse.update(wilson95=(low, high), compatible_with_half=low <= .50 <= high)
    passed = (all(metrics[d]["teacher"]["rate"] >= .95 for d in DISTANCES)
              and all(metrics[d][name]["rate"] < .10 for d in (30,60,100)
                      for name in ("turn_180", "scan", "find_structure", "start_pose"))
              and all(metrics[d]["reverse_route"]["compatible_with_half"] for d in (30,60,100)))
    save(output / "metrics.json", dict(passed=passed, distances=metrics))
    save(output / "fixtures.json", dict(protocol=PROTOCOL, formal=ACCEPTANCE_SEEDS, fixtures=configs))
    with (output / "acceptance.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=next(iter(rows.values())).keys())
        writer.writeheader()
        writer.writerows(rows.values())
    if not passed:
        raise RuntimeError("T1 v3 teacher or RGB/script shortcut acceptance failed")


def build_index(collection, frozen, previous, previous_fixtures, output):
    output.mkdir(parents=True, exist_ok=False)
    groups = {}
    all_seeds, fingerprints = set(), set()
    old_scenes, old_worlds = set(), set()
    for source in previous:
        for line in source.read_text().splitlines():
            row = json.loads(line)
            old_scenes.add(row["scene_seed"])
            old_worlds.add(row["world_seed"])
    for source in (frozen, *previous_fixtures):
        fixtures = json.loads(source.read_text())["fixtures"]
        old_scenes.update(entry["scenario_seed"] for entry in fixtures.values())
        old_worlds.update(entry["world_seed"] for entry in fixtures.values())
    world_sets = {}
    for name in ("train", "validation", "test", "dart"):
        rows = read_episodes([collection / name], protocol=PROTOCOL)
        split = "train" if name == "dart" else name
        if not rows or any(row["split"] != split for row in rows):
            raise ValueError(f"Invalid {name} collection")
        current_seeds = {row["scene_seed"] for row in rows}
        current_fingerprints = {row["fingerprint"] for row in rows}
        current_worlds = {row["world_seed"] for row in rows}
        if current_seeds & (old_scenes | all_seeds) or current_fingerprints & fingerprints or current_worlds & old_worlds:
            raise ValueError(f"Duplicate or benchmark-overlapping trajectory in {name}")
        all_seeds.update(current_seeds)
        fingerprints.update(current_fingerprints)
        world_sets[name] = current_worlds
        quotas = DART_QUOTAS if name == "dart" else TRAIN_QUOTAS if name == "train" else HELD_OUT_QUOTAS
        groups[name] = [dict(row, role=name) for row in select(rows, split, quotas)]
    if sum(map(len, world_sets.values())) != len(set().union(*world_sets.values())):
        raise ValueError("Collection world seeds overlap")
    frames, labels = Counter(), Counter()
    path = output / "dataset.jsonl"
    with path.open("x") as stream:
        for name, rows in groups.items():
            for row in rows:
                key = f"{name}/{row['group']}"
                frames[key] += row["selected_frames"]
                labels[key] += row["selected_labels"]
                stream.write(json.dumps(row)+"\n")
    save(output / "summary.json", dict(protocol=PROTOCOL, frame_counts=frames, label_counts=labels,
                                        index_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                        episodes={name:len(rows) for name,rows in groups.items()},
                                        world_seeds={name:sorted(worlds) for name,worlds in world_sets.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("acceptance")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    index = subs.add_parser("index")
    index.add_argument("--collection", type=Path, required=True)
    index.add_argument("--frozen", type=Path, required=True)
    index.add_argument("--previous", type=Path, nargs="+", required=True)
    index.add_argument("--previous-fixtures", type=Path, nargs="*", default=[])
    index.add_argument("--output", type=Path, required=True)
    incomplete = subs.add_parser("incomplete-shards")
    incomplete.add_argument("--input", type=Path, required=True)
    incomplete.add_argument("--split", choices=("train", "validation", "test", "dart"), required=True)
    incomplete.add_argument("--workers", type=int, choices=range(1, 33), required=True)
    args = parser.parse_args()
    if args.command == "acceptance":
        acceptance(args.input, args.output)
    elif args.command == "incomplete-shards":
        print(*incomplete_shards(args.input, args.split, args.workers))
    else:
        build_index(args.collection, args.frozen, args.previous, args.previous_fixtures, args.output)
