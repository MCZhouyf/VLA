"""Locate screened T1 death and respawn events in the archived CPU traces."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


def position(row):
    return tuple(float(row[k]) for k in ("x", "y", "z"))


def box_distance(point, box, axes):
    return math.sqrt(sum(max(box[i] - point[i], point[i] - box[i + 3] - 1, 0) ** 2 for i in axes))


def summarize(scores):
    with scores.open(newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        if row["d"] != "100":
            continue
        for policy, column, tag in (("turn180", "turn180_deaths", "180"), ("rgb_scan", "deaths", "rgb")):
            expected = int(row[column])
            if not expected:
                continue
            folder = (scores.parent / row["source"]).parent
            stem = f"d100-seed{row['seed']}"
            config = json.loads((folder / f"{stem}.json").read_text())
            path = folder / f"{stem}-{tag}.csv.gz"
            with gzip.open(path, "rt", newline="") as stream:
                trace = list(csv.DictReader(stream))
            events = [index for index, step in enumerate(trace) if step["death"] == "True"]
            if len(events) != expected:
                raise ValueError(f"Death count differs from archived trace: {path}")
            ax, floor, az = config["arena_origin"]
            boxes = [*config["occlusion_walls"], *config["pillar_screen"]]
            for index in events:
                if index == 0:
                    raise ValueError(f"Death has no preceding position: {path}")
                before, dead = trace[index - 1:index + 1]
                x, y, z = final = position(before)
                fatal_prefix = trace[:index]
                outside = [step for step in fatal_prefix
                           if not (ax - 110 <= float(step["x"]) < ax + 111
                                   and az - 110 <= float(step["z"]) < az + 111)]
                below = [step for step in fatal_prefix if float(step["y"]) < floor - .6]
                below_after_exit = [step for step in below
                                    if outside and int(step["step"]) >= int(outside[0]["step"])]
                respawn = position(dead)
                expected_respawn = config["respawn_point"]
                respawn_match = math.dist(respawn, expected_respawn) <= .8
                yield dict(
                    distance=100, seed=row["seed"], world_seed=row["world_seed"], policy=policy,
                    death_step=dead["step"], death_counter_increased=int(dead["death_count"]) > int(before["death_count"]),
                    pose_jump=dead["pose_jump"], first_outside_step=outside[0]["step"] if outside else "",
                    first_below_floor_step=below[0]["step"] if below else "",
                    first_below_floor_after_exit_step=below_after_exit[0]["step"] if below_after_exit else "",
                    arena_y=floor,
                    before_x=x, before_y=y, before_z=z, below_floor=floor-y,
                    nearest_structure_horizontal=min(box_distance(final, box, (0, 2)) for box in boxes),
                    nearest_structure_3d=min(box_distance(final, box, range(3)) for box in boxes),
                    health_before=before["health"], health_after=dead["health"],
                    respawn_x=respawn[0], respawn_y=respawn[1], respawn_z=respawn[2],
                    matches_respawn_pad=respawn_match,
                    trace=str(path.relative_to(scores.parent)),
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(summarize(args.scores))
    if not rows:
        raise ValueError("No screened death events found")
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
