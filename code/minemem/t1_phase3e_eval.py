"""Run the frozen first-arrival T1 v3 oracle-waypoint controller evaluation."""

import argparse
import csv
import json
import math
from copy import copy
from pathlib import Path
from time import perf_counter

from t1_dual_enclosure import configure_rendering
from t1_phase3e import DISTANCES, PROTOCOL, FirstArrivalArena
from t1_respawn import record_config
from t1_safety import StepTrace
from t1_trained_eval import ControlTrace, Controller, drive_rollouts, rollout
from t1_waypoint import (STABILIZER_ALTERNATING_TURNS, STABILIZER_FORWARD_STEPS,
                         RouteLookahead, WaypointTurnStabilizer)


GEOMETRY = ("protocol", "scenario_seed", "world_seed", "goal_slot", "candidate_positions", "walls",
            "spawn", "return_pose", "teacher_targets", "distractor")
FIELDS = ("d", "seed", "world_seed", "success", "steps", "final_dist", "min_dist", "decoded_steps",
          "elapsed_seconds", "deaths", "pose_jump", "pose_jumps", "end_reason", "return_limit",
          "left_turn_steps", "right_turn_steps", "turn_in_place_steps", "horizontal_path")


def movement_summary(trace):
    left = right = stationary_turns = 0
    travelled = 0.0
    for previous, current in zip(trace.rows, trace.rows[1:]):
        distance = math.dist((previous[1], previous[3]), (current[1], current[3]))
        yaw = (current[4]-previous[4]+180) % 360-180
        travelled += distance
        left += yaw < -.01
        right += yaw > .01
        stationary_turns += abs(yaw) > .01 and distance < .02
    return left, right, stationary_turns, travelled


def checked_scene(arena, expected):
    seed, distance = expected["scenario_seed"], expected["distance"]
    config = arena.setup(seed, distance, expected["goal_slot"])
    for key in GEOMETRY:
        if json.loads(json.dumps(config[key])) != expected[key]:
            raise RuntimeError(f"Frozen v3 geometry mismatch for d{distance}-seed{seed}: {key}")
    return config


def run_seed(args, cases, controller, emit):
    worlds = {(case["world_seed"], case["optional_effects"]) for case in cases}
    if len(worlds) != 1:
        raise ValueError("All distances in one rollout lane must use the same frozen world and effects")
    arena = FirstArrivalArena(*worlds.pop())
    try:
        staged = []
        for expected in cases:
            config = checked_scene(arena, expected)
            tag = f"d{expected['distance']}-seed{expected['scenario_seed']}"
            out_trace = StepTrace(args.output / f"{tag}-outbound.csv.gz")
            arena.capture = []
            try:
                arena.outward(config, out_trace)
                staged.append((expected, tag, arena.capture[-controller.policy.history:]))
            finally:
                arena.capture = None
                arena.distractor = None
        for expected, tag, outbound in staged:
            checked_scene(arena, expected)
            arena.landmark, arena.distractor = expected["landmark"], expected["distractor"]
            record_config(args.output, tag, expected)
            start = expected["return_pose"]
            route = RouteLookahead((start["x"], start["z"]), expected["teacher_targets"])

            def instruction(info):
                poses = info["player_pos"]
                return route.condition([float(poses[key]) for key in ("x", "z", "yaw")])

            trace = ControlTrace(args.output / f"{tag}-return.csv.gz", args.output / f"{tag}-actions.jsonl")
            limit = expected["return_limit"]
            stabilizer = WaypointTurnStabilizer() if args.stabilize_turns else None
            result = yield from rollout(arena, controller, expected, outbound, instruction, limit, trace,
                                        stabilize_turns=stabilizer)
            emit((expected["distance"], expected["scenario_seed"], expected["world_seed"],
                  *result, limit, *movement_summary(trace)))
            arena.distractor = None
    finally:
        arena.sim.close()


def run(args):
    metrics = json.loads(args.metrics.read_text())
    fixtures = json.loads(args.fixtures.read_text())
    if not metrics["passed"] or fixtures["protocol"] != PROTOCOL:
        raise ValueError("The T1 v3 benchmark must pass CPU scene acceptance")
    controller = Controller(args.checkpoint, args.training_run)
    if controller.settings["variant"] != "no_memory" or controller.settings.get("evaluation_history") != "placeholder":
        raise ValueError("Use the frozen waypoint controller with placeholder action history")
    configure_rendering(args.output)
    seeds = args.seeds or fixtures["formal"]
    if not set(seeds) <= set(fixtures["formal"]) or len(set(seeds)) != len(seeds):
        raise ValueError("Select distinct frozen evaluation seeds")
    cases = [[fixtures["fixtures"][f"{distance}/{seed}"] for distance in args.distances]
             for seed in seeds]
    if args.stabilize_turns or args.seeds:
        (args.output / "control_protocol.json").write_text(json.dumps(dict(
            weights=str(args.training_run), stabilize_turns=args.stabilize_turns,
            turn_stabilizer=dict(alternating_turns=STABILIZER_ALTERNATING_TURNS,
                                 forward_steps=STABILIZER_FORWARD_STEPS) if args.stabilize_turns else None,
            provenance="targeted_diagnostic" if args.seeds else "full_stabilized_recheck",
            distances=args.distances, seeds=seeds,
        ), indent=2)+"\n")
    with (args.output / "episodes.csv").open("x", newline="") as stream, (
            args.output / "rollout_timing.csv").open("x", newline="") as timing:
        writer, timings = csv.writer(stream), csv.writer(timing)
        writer.writerow(FIELDS)
        timings.writerow(("episodes", "steps", "wall_seconds", "aggregate_fps"))
        for offset in range(0, len(cases), args.batch):
            batch = cases[offset:offset+args.batch]
            records = []

            def emit(row):
                writer.writerow(row)
                stream.flush()
                records.append(row)

            started = perf_counter()
            drive_rollouts([run_seed(args, case, copy(controller), emit) for case in batch], lambda *_: None)
            elapsed = perf_counter()-started
            steps = sum(row[4] for row in records)
            timings.writerow((len(records), steps, elapsed, steps/elapsed))
            timing.flush()
            print(f"done={(offset+len(batch))*len(args.distances)}/{len(cases)*len(args.distances)} "
                  f"steps={steps} fps={steps/elapsed:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "checkpoint", "training-run", "fixtures", "metrics"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--distances", type=int, nargs="+", choices=DISTANCES, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--stabilize-turns", action="store_true")
    parser.add_argument("--batch", type=int, default=3)
    args = parser.parse_args()
    if args.batch < 1 or len(set(args.distances)) != len(args.distances):
        parser.error("Invalid rollout batch or duplicated distance")
    run(args)
