"""Verify the paired Phase 3 D30 result directly from the frozen poses."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import median

from phase1_metrics import wilson


GEOMETRY = ("protocol", "world_seed", "scenario_seed", "distance", "arena_origin", "landmark",
            "distractors", "wall", "waypoints", "planned_endpoint", "occlusion_walls", "pillar_screen",
            "screened_pillar", "render_distance_chunks", "return_limit", "return_pose", "respawn_point")


def audit_episode(folder, row, fixture):
    seed = int(row["seed"])
    tag = f"d30-seed{seed}"
    config = json.loads((folder / f"{tag}.json").read_text())
    if any(config[key] != fixture[key] for key in GEOMETRY):
        raise ValueError(f"Scene geometry differs from the frozen case for {folder.name}/{seed}")
    with gzip.open(folder / f"{tag}-return.csv.gz", "rt", newline="") as stream:
        trace = list(csv.DictReader(stream))
    with gzip.open(folder / f"{tag}-outbound.csv.gz", "rt", newline="") as stream:
        outbound = list(csv.DictReader(stream))
    steps = len(trace) - 1
    pose = [tuple(float(point[key]) for key in ("x", "z", "yaw")) for point in trace]
    initial_pose = (fixture["return_pose"]["x"], fixture["return_pose"]["z"])
    distances = [math.dist(point[:2], fixture["landmark"]) for point in pose]
    deaths = sum(point["death"] == "True" for point in trace)
    jumps = sum(point["pose_jump"] == "True" for point in trace)
    expected_success = min(distances) <= 3.0 and deaths == jumps == 0
    if (not outbound or steps != int(row["steps"]) or steps != int(row["decoded_steps"])
            or int(row["return_limit"]) != fixture["return_limit"]
            or (row["end_reason"] == "budget" and steps != fixture["return_limit"])
            or math.dist(initial_pose, pose[0][:2]) > 1e-4
            or not math.isclose(min(distances), float(row["min_dist"]), abs_tol=1e-6)
            or not math.isclose(distances[-1], float(row["final_dist"]), abs_tol=1e-6)
            or not math.isclose(config["path_len_out"], float(row["path_len_out"]), abs_tol=1e-6)
            or deaths != int(row["deaths"]) or jumps != int(row["pose_jumps"])
            or expected_success != (row["success"] == "True")):
        raise ValueError(f"Trajectory differs from episode metadata for {folder.name}/{seed}")
    yaw = [(after[2] - before[2] + 180) % 360 - 180 for before, after in zip(pose, pose[1:])]
    return dict(seed=seed, success=expected_success, steps=steps, deaths=deaths, pose_jumps=jumps,
                min_dist=min(distances), path=sum(math.dist(a[:2], b[:2]) for a, b in zip(pose, pose[1:])),
                left_turn_steps=sum(angle < -0.01 for angle in yaw),
                right_turn_steps=sum(angle > 0.01 for angle in yaw))


def report(run, fixtures):
    frozen = json.loads(fixtures.read_text())
    seeds = set(frozen["formal"])
    checked = {}
    arms = {}
    for name in ("no_memory", "spatial"):
        folder = run / name
        with (folder / "episodes.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if (len(rows) != len(seeds) or {int(row["seed"]) for row in rows} != seeds
                or any(row["d"] != "30" or row["mode"] != "formal" or row["model"] != name for row in rows)):
            raise ValueError(f"Incorrect frozen seed coverage or experiment mode for {name}")
        checked[name] = {int(row["seed"]): audit_episode(folder, row, frozen["fixtures"][f"30/{row['seed']}"])
                         for row in rows}
        results = list(checked[name].values())
        wins = sum(row["success"] for row in results)
        steps = sum(row["steps"] for row in results)
        with (folder / "rollout_timing.csv").open(newline="") as stream:
            wall_seconds = sum(float(row["wall_seconds"]) for row in csv.DictReader(stream))
        arms[name] = dict(episodes=len(results), successes=wins, wilson95=wilson(wins, len(results)),
                          steps=steps, deaths=sum(row["deaths"] for row in results),
                          pose_jumps=sum(row["pose_jumps"] for row in results),
                          min_distance_median=median(row["min_dist"] for row in results),
                          best_distance=min(row["min_dist"] for row in results),
                          return_path_median=median(row["path"] for row in results),
                          left_turn_steps=sum(row["left_turn_steps"] for row in results),
                          right_turn_steps=sum(row["right_turn_steps"] for row in results),
                          wall_seconds=wall_seconds, aggregate_return_fps=steps / wall_seconds)
    differences = [checked["spatial"][seed]["min_dist"] - checked["no_memory"][seed]["min_dist"]
                   for seed in seeds]
    paired = dict(spatial_closer=sum(value < -1e-3 for value in differences),
                  no_memory_closer=sum(value > 1e-3 for value in differences),
                  tied=sum(abs(value) <= 1e-3 for value in differences),
                  spatial_minus_no_memory_median=median(differences))
    decision = json.loads((run / "early-decision.json").read_text())
    if (decision["formal_d30_successes"] != {name: result["successes"] for name, result in arms.items()}
            or decision["episodes_per_variant"] != len(seeds)
            or decision["proceed"] != (arms["no_memory"]["successes"] < 6 and arms["spatial"]["successes"] > 15)):
        raise ValueError("The coordinator decision differs from the audited episodes")
    return dict(arms=arms, paired_min_distance=paired, decision=decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.run, args.fixtures), indent=2))
