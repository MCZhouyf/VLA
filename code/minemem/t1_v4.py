"""T1 randomized return headings, teacher acceptance and shortcut controls."""

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np

from phase1_t1 import (
    Arena, DISTANCES, POINT_TOLERANCE, SUCCESS_RADIUS, TEACHER_SEED,
    WORLD_COUNT, WORLD_REPLACEMENTS, WORLD_SEED, bearing, horizontal, wrapped_angle,
)


PROTOCOL = "t1_v4"
INSTRUCTION = "Go back to the gold pillar you saw."
VISIBLE_INSTRUCTION = "Go to the gold block."


class RandomHeadingArena(Arena):
    def prepare_episode(self, scenario_seed, distance):
        # A shortcut or a model can have walked past the arena between episodes.
        center_x, center_z = self.landmark
        self.command(f"/tp @p {center_x + 2} {self.y} {center_z} 0 0")
        self.settle()
        config = super().prepare_episode(scenario_seed, distance)
        config["protocol"] = PROTOCOL
        return config

    def outbound_segment(self, config):
        for _ in range(40):
            self.teacher_step(record=True)
        for target in config["waypoints"]:
            _, reached, _ = self.drive(target, POINT_TOLERANCE, 800, record=True)
            if not reached:
                raise RuntimeError(f"Teacher could not reach outbound waypoint {target}")
        for _ in range(8):
            self.teacher_step(record=True)
        goal_yaw = bearing(horizontal(self.info), self.landmark)
        rng = random.Random(f"{PROTOCOL}:{config['scenario_seed']}:{config['distance']}")
        offset = rng.uniform(60, 300)
        yaw = wrapped_angle(goal_yaw + offset)
        self.turn_to(yaw, record=True)
        for _ in range(20):
            self.teacher_step(record=True)
        pose = self.info["player_pos"]
        actual = math.dist(horizontal(self.info), self.landmark)
        separation = abs(wrapped_angle(pose["yaw"] - bearing(horizontal(self.info), self.landmark)))
        if abs(actual - config["distance"]) > POINT_TOLERANCE + 0.2 or separation < 60:
            raise RuntimeError(f"T1 endpoint check failed: distance={actual}, heading offset={separation}")
        config.update(return_pose=pose, actual_distance=actual, path_len_out=self.path_len_out,
                      final_yaw=yaw, sampled_heading_offset=offset,
                      actual_heading_separation=separation)

    def turn_to(self, yaw, record=False):
        for step in range(30):
            error = wrapped_angle(yaw - self.info["player_pos"]["yaw"])
            if abs(error) <= 0.01:
                return step
            self.teacher_step(yaw_delta=max(-10, min(10, error)), record=record)
        raise RuntimeError("Turn did not converge")

    def shortcut(self, fixture):
        self.restore_return_pose(fixture)
        initial = self.info["player_pos"]["yaw"]
        # Use the live heading only to turn 180 degrees. The goal position is
        # consumed by the evaluator to score success, never to choose controls.
        steps = self.turn_to(wrapped_angle(initial + 180))
        closest = math.dist(horizontal(self.info), self.landmark)
        action = self.sim.noop_action()
        action["forward"] = 1
        for steps in range(steps + 1, fixture["return_limit"] + 1):
            if not self.navigation_step(action.copy()):
                return False, steps, closest, self.navigation_end
            closest = min(closest, math.dist(horizontal(self.info), self.landmark))
            if closest <= SUCCESS_RADIUS:
                return True, steps, closest, "reached"
        return False, steps, closest, "budget"

    def navigation_step(self, action):
        previous = horizontal(self.info)
        _, _, ended, truncated, self.info = self.sim.step(action)
        current = horizontal(self.info)
        if not all(map(math.isfinite, current)) or math.dist(previous, current) > 2:
            self.navigation_end = f"discontinuous pose {previous} -> {current}"
            return False
        if ended or truncated or not self.info.get("life_stats", {}).get("is_alive", True):
            self.navigation_end = "terminated"
            return False
        return True


def evaluate_case(arena, old, output, recheck=None):
    d, seed = old["distance"], old["scenario_seed"]
    scene = arena.prepare_episode(seed, d)
    for key in ("arena_origin", "wall", "landmark", "waypoints"):
        if json.loads(json.dumps(scene[key])) != old[key]:
            raise RuntimeError(f"Frozen T1 scene changed at {d}/{seed}: {key}")
    # Always use the archived v3 start and reference budget in the old control.
    old_shortcut = arena.shortcut(old)
    scene = arena.prepare_episode(seed, d)
    key = f"d{d}-seed{seed}"
    if recheck is None:
        arena.outbound_segment(scene)
        teacher_ok, steps, final, closest = arena.teacher_return(scene)
        if not teacher_ok:
            raise RuntimeError(f"Teacher did not solve {key}, closest={closest}")
        with (output / f"{key}-outbound.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "z", "yaw", "action_token"))
            writer.writerows(arena.outbound)
    else:
        original, path = recheck[d, seed]
        scene = json.loads((path / f"{key}.json").read_text())
        if (original["error"] or scene["protocol"] != PROTOCOL
                or scene["world_seed"] != arena.world_seed):
            raise RuntimeError(f"No valid archived T1 v4 teacher for {key}")
        steps, final, closest = (original[column] for column in
                                 ("teacher_steps", "teacher_final_dist", "teacher_min_dist"))
        shutil.copyfile(path / f"{key}-outbound.csv", output / f"{key}-outbound.csv")
    new_shortcut = arena.shortcut(scene)
    (output / f"{key}.json").write_text(json.dumps(scene, indent=2, default=float) + "\n")
    return (d, seed, scene["world_seed"], steps, final, closest, scene["path_len_out"],
            scene["actual_heading_separation"], *old_shortcut, *new_shortcut, "")


ACCEPTANCE_FIELDS = (
    "distance", "seed", "world_seed", "teacher_steps", "teacher_final_dist",
    "teacher_min_dist", "path_len_out", "heading_separation",
    "v3_shortcut_success", "v3_shortcut_steps", "v3_shortcut_min_dist", "v3_shortcut_end",
    "v4_shortcut_success", "v4_shortcut_steps", "v4_shortcut_min_dist", "v4_shortcut_end", "error",
)


def assigned_seeds(worker, workers):
    for world in range(worker, WORLD_COUNT, workers):
        yield WORLD_REPLACEMENTS.get(world, WORLD_SEED + world), [
            TEACHER_SEED + index for index in range(world, 50, WORLD_COUNT)
        ]


def acceptance(args):
    from phase1_t1_eval import fixture_index

    fixtures = fixture_index(args.legacy)
    verified = set()
    if args.skip_verified:
        for source in args.skip_verified.glob("worker-*/acceptance.csv"):
            with source.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if not row["error"] and row.get("v3_shortcut_end") and row.get("v4_shortcut_end"):
                        verified.add((int(row["distance"]), int(row["seed"])))
    recheck = None
    if args.recheck:
        recheck = {}
        for source in args.recheck.glob("worker-*/acceptance.csv"):
            with source.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    key = int(row["distance"]), int(row["seed"])
                    if key in recheck:
                        raise ValueError(f"Duplicate archived T1 case {key}")
                    recheck[key] = (row, source.parent)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "acceptance.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(ACCEPTANCE_FIELDS)
        stream.flush()
        for world_seed, assigned in assigned_seeds(args.worker, args.workers):
            cases = [(d, seed) for seed in assigned for d in DISTANCES if (d, seed) not in verified]
            if not cases:
                continue
            arena = RandomHeadingArena(world_seed)
            try:
                for d, seed in cases:
                    try:
                        row = evaluate_case(arena, fixtures[d, seed], args.output, recheck)
                    except (RuntimeError, OSError) as error:
                        row = (d, seed, world_seed, *("",) * (len(ACCEPTANCE_FIELDS) - 4), str(error))
                    writer.writerow(row)
                    stream.flush()
                    print(row, flush=True)
            finally:
                arena.sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--legacy", nargs="+", required=True, type=Path)
    parser.add_argument("--recheck", type=Path, help="Reuse a finished v4 teacher run and replay both shortcuts")
    parser.add_argument("--skip-verified", type=Path, help="Replay only cases missing from a previous shortcut run")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=WORLD_COUNT)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers <= WORLD_COUNT:
        parser.error("Invalid simulator shard")
    acceptance(args)


if __name__ == "__main__":
    main()
