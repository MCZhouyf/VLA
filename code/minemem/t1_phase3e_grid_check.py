"""Check both routes in each archived 30-block scene with a CPU heading-and-step proxy."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import median

from t1_phase3e_closed_audit import large_turn_switches
from t1_phase3e_grid_plan import GridRoute
from t1_waypoint import waypoint_labels


def check_case(config, poses, slot):
    planner = GridRoute(poses, slot)
    candidates = config["candidate_positions"]
    x, z, yaw = (config["return_pose"][key] for key in ("x", "z", "yaw"))
    initial = planner.target((x, z, yaw))
    invariant = all(planner.target((x, z, angle)) == initial for angle in range(0, 360, 10))
    outcome = "budget"
    for step in range(config["return_limit"] + 1):
        if math.dist((x, z), candidates[1-slot]) <= 3:
            outcome = "other_room_first"
            break
        if math.dist((x, z), candidates[slot]) <= 3:
            outcome = "reached"
            break
        if step == config["return_limit"]:
            break
        direction, distance = waypoint_labels((x, z, yaw), planner.target((x, z, yaw)))
        if direction:
            yaw += 10 if direction <= 18 else -10
        elif distance:
            x -= .18 * math.sin(math.radians(yaw))
            z += .18 * math.cos(math.radians(yaw))
        else:
            outcome = "zero_distance_command"
            break
    return dict(seed=config["scenario_seed"], slot=slot, steps=step, outcome=outcome,
                observed_goal_distance=math.dist(planner.goal, candidates[slot]),
                initial_target_yaw_invariant=invariant)


def audit(root):
    rows = []
    arm = root / "spatial" / "attempt-1"
    for path in sorted(arm.glob("d30-seed*-outbound.csv.gz")):
        tag = path.name.removesuffix("-outbound.csv.gz")
        config = json.loads((arm / f"{tag}.json").read_text())
        with gzip.open(path, "rt") as stream:
            poses = [tuple(float(row[key]) for key in ("x", "z", "yaw")) for row in csv.DictReader(stream)]
        rows.extend(check_case(config, poses, slot) for slot in (0, 1))
    frozen = set(range(2086000000, 2086000030))
    if {(row["seed"], row["slot"]) for row in rows} != {(seed, slot) for seed in frozen for slot in (0,1)}:
        raise ValueError("The archived outbound traces do not cover both slots of all 30 frozen scenes")
    prior = {}
    for name in ("no_memory", "long", "spatial"):
        with (root / name / "episodes.csv").open(newline="") as stream:
            episodes = list(csv.DictReader(stream))
        if len(episodes) != 30 or {int(row["seed"]) for row in episodes} != frozen:
            raise ValueError(f"Invalid archived seed coverage in {name}")
        prior[name] = dict(median_min_dist=median(float(row["min_dist"]) for row in episodes),
                           median_final_dist=median(float(row["final_dist"]) for row in episodes),
                           reached_within_12=sum(float(row["min_dist"]) <= 12 for row in episodes))
        if name == "spatial":
            switches = []
            for row in episodes:
                with (arm / f"d30-seed{row['seed']}-actions.jsonl").open() as stream:
                    count = large_turn_switches([json.loads(line) for line in stream])
                switches.append(dict(seed=int(row["seed"]), success=row["success"] == "True", switches=count))
            prior[name]["waypoint_turn_switches_tail200"] = switches
    return dict(proxy="10-degree rotations and 0.18-block forward steps, given each candidate slot",
                scenes=len(frozen), routes=len(rows), successes=sum(row["outcome"] == "reached" for row in rows),
                yaw_invariant=sum(row["initial_target_yaw_invariant"] for row in rows),
                rows=rows, prior=prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.closed)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"CPU proxy reached {result['successes']}/{result['routes']} selected destinations")
