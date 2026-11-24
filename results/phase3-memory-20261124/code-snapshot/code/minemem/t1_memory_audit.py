"""Audit spatial-grid retention on the frozen T1 outbound pose traces."""

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from struct import pack, unpack


CELL_BLOCKS = 2
HEADINGS = 8
GRID_SIDES = (32, 64, 128)
ARRIVAL_RADIUS = 3
OUTBOUND_SUFFIX = "-outbound.csv.gz"


def fp32(value):
    return unpack("f", pack("f", value))[0]


def retained_indices(poses, side):
    cells = [(math.floor(fp32(x) / CELL_BLOCKS), math.floor(fp32(z) / CELL_BLOCKS))
             for x, z, _ in poses]
    origin = tuple(index - side // 2 for index in cells[0])
    origins = []
    for cell in cells:
        origin = tuple(max(index - side + 1, min(before, index))
                       for before, index in zip(origin, cell, strict=True))
        origins.append(origin)
    lower, upper = origins[-1], tuple(origin + side - 1 for origin in origins[-1])
    kept = set()
    for index in range(len(cells) - 2, -1, -1):
        lower = tuple(max(a, b) for a, b in zip(lower, origins[index], strict=True))
        upper = tuple(min(a, b + side - 1) for a, b in zip(upper, origins[index], strict=True))
        if all(a <= cell <= b for a, cell, b in zip(lower, cells[index], upper, strict=True)):
            kept.add(index)
    return cells, kept


def matching_traces(fixtures, trace_roots):
    expected = {f"d{entry['distance']}-seed{entry['scenario_seed']}": entry for entry in fixtures}
    matched = {}
    for root in trace_roots:
        for path in sorted(root.rglob(f"*{OUTBOUND_SUFFIX}")):
            name = path.name.removesuffix(OUTBOUND_SUFFIX)
            if name not in expected or name in matched:
                continue
            config = path.with_name(f"{name}.json")
            if config.is_file() and json.loads(config.read_text()) == expected[name]:
                matched[name] = path, expected[name]
    if len(matched) != len(expected):
        raise ValueError(f"Missing exact frozen pose traces: {sorted(expected.keys() - matched.keys())}")
    return matched


def inspect_trace(path, fixture):
    with gzip.open(path, "rt") as stream:
        poses = [(float(row["x"]), float(row["z"]), float(row["yaw"])) for row in csv.DictReader(stream)]
    dwell = fixture["outbound_dwell_steps"]
    if len(poses) <= dwell:
        raise ValueError(f"Incomplete outbound trace: {path}")
    recorded_end = fixture["return_pose"]
    # Captured observation i is the pre-action state. The last pose is the
    # restored state at which the model first gets to query its history.
    poses[-1] = tuple(float(recorded_end[key]) for key in ("x", "z", "yaw"))
    initial_distance = math.dist(poses[0][:2], fixture["landmark"])
    result = dict(seed=fixture["scenario_seed"], distance=fixture["distance"],
                  outbound_frames=len(poses)-1, anchor_frames=dwell,
                  start_to_gold=initial_distance, start_within_success_radius=initial_distance <= ARRIVAL_RADIUS,
                  trace=str(path))
    headings = [math.floor((fp32(yaw) % 360) * HEADINGS / 360) for _, _, yaw in poses[:-1]]
    for side in GRID_SIDES:
        width = side * CELL_BLOCKS
        cells, kept = retained_indices(poses, side)
        anchors = kept.intersection(range(dwell))
        initial_slots = {(cells[i], headings[i]) for i in anchors}
        later_overwrites = sum((cells[i], headings[i]) in initial_slots for i in kept if i >= dwell)
        result.update({f"total_kept_{width}": len(kept), f"anchor_kept_{width}": len(anchors),
                       f"same_slot_later_observations_{width}": later_overwrites})
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["distance"]].append(row)
    summary = {}
    for distance, cases in sorted(groups.items()):
        summary[distance] = dict(episodes=len(cases), recorded_start_within_3=sum(
            row["start_within_success_radius"] for row in cases), grids={})
        for side in GRID_SIDES:
            width = side * CELL_BLOCKS
            retained = [row for row in cases if row[f"anchor_kept_{width}"]]
            summary[distance]["grids"][width] = dict(
                episodes_with_anchor=len(retained),
                episodes_retaining_entire_dwell=sum(
                    row[f"anchor_kept_{width}"] == row["anchor_frames"] for row in cases),
                retained_anchor_episodes_with_later_same_slot_writes=sum(
                    row[f"same_slot_later_observations_{width}"] > 0 for row in retained))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--trace-roots", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())["fixtures"]
    matched = matching_traces(fixtures.values(), args.trace_roots)
    rows = [inspect_trace(path, fixture) for path, fixture in matched.values()]
    rows.sort(key=lambda row: (row["distance"], row["seed"]))
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
