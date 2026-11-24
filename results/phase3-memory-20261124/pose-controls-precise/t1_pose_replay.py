"""Evaluate return policies that store only their outbound poses."""

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from t1_collect import RevisedCaptureArena
from t1_occlusion import SCREENED_PROTOCOL, set_candidate_render_distance
from t1_respawn import GEOMETRY, freeze_return, record_config
from t1_safety import StepTrace, TrialInterrupted
from t1_v4 import DISTANCES, SUCCESS_RADIUS, bearing, horizontal, wrapped_angle


CONTROLS = ("start_pose", "reverse_poses")
FIELDS = ("d", "seed", "world_seed", "control", "success", "steps", "return_limit", "min_dist",
          "final_dist", "deaths", "pose_jump", "pose_jumps", "end_reason", "stored_waypoints", "error")


def recorded_route(rows, spacing=1.3):
    poses = [(row[1], row[3]) for row in rows]
    route = [poses[-1]]
    for point in reversed(poses[:-1]):
        if math.dist(point, route[-1]) >= spacing:
            route.append(point)
    if route[-1] != poses[0]:
        route.append(poses[0])
    return poses[0], route[1:]


def return_by_pose(arena, fixture, targets, trace, terminal_tolerance):
    arena.restore_return_pose(fixture)
    target_queue = deque(targets)
    steps, stalled = 0, 0
    closest = final = math.dist(horizontal(arena.info), arena.landmark)
    end = "budget"
    try:
        with arena.track(trace):
            for _ in range(fixture["return_limit"]):
                position = horizontal(arena.info)
                while target_queue:
                    tolerance = terminal_tolerance if len(target_queue) == 1 else 0.4
                    if math.dist(position, target_queue[0]) > tolerance:
                        break
                    target_queue.popleft()
                if not target_queue:
                    end = "stored_pose_reached"
                    break
                yaw = arena.info["player_pos"]["yaw"]
                error = wrapped_angle(bearing(position, target_queue[0])-yaw)
                forward = abs(error) <= 8
                action = arena.sim.noop_action()
                final_approach = (terminal_tolerance < 0.4 and len(target_queue) == 1
                                  and math.dist(position, target_queue[0]) < 0.75)
                action.update(camera=np.array([0.0, max(-10, min(10, error))], dtype=np.float32),
                              forward=int(forward), sneak=int(final_approach),
                              jump=int(forward and stalled >= 5 and stalled % 6 == 5))
                steps += 1
                if not arena.navigation_step(action):
                    break
                current = horizontal(arena.info)
                stalled = stalled+1 if forward and math.dist(position, current) < 0.02 else 0
                final = math.dist(current, arena.landmark)
                closest = min(closest, final)
                if closest <= SUCCESS_RADIUS:
                    end = "reached"
                    break
    except TrialInterrupted:
        pass
    success = closest <= SUCCESS_RADIUS and not trace.reason
    return success, steps, fixture["return_limit"], closest, final, *trace.flags(), trace.reason or end


def run_case(arena, expected, output, terminal_tolerance):
    seed, distance = expected["scenario_seed"], expected["distance"]
    config = arena.prepare_episode(seed, distance)
    for key in (*GEOMETRY, "world_seed", "scenario_seed", "protocol", "occlusion_walls",
                "pillar_screen", "screened_pillar", "render_distance_chunks"):
        if json.loads(json.dumps(config.get(key))) != expected.get(key):
            raise RuntimeError(f"Frozen pose-control scene differs in {key}")
    if config["protocol"] != SCREENED_PROTOCOL:
        raise ValueError("Unexpected pose-control scene protocol")
    tag = f"d{distance}-seed{seed}"
    trace = StepTrace(output / f"{tag}-outbound.csv.gz")
    with arena.track(trace):
        arena.outbound_segment(config)
    freeze_return(config, expected)
    config["pose_only_terminal_tolerance"] = terminal_tolerance
    record_config(output, tag, config)
    start, route = recorded_route(trace.rows)
    targets = {"start_pose": [start], "reverse_poses": route}
    arena.command("/gamemode adventure @p")
    arena.settle()
    try:
        for name, waypoints in targets.items():
            return_trace = StepTrace(output / f"{tag}-{name}.csv.gz")
            result = return_by_pose(arena, config, waypoints, return_trace, terminal_tolerance)
            yield distance, seed, expected["world_seed"], name, *result, len(waypoints), ""
    finally:
        arena.command("/gamemode survival @p")


def run(args):
    fixture_file = json.loads(args.fixtures.read_text())
    seeds = fixture_file["formal"]
    fixtures = {(entry["scenario_seed"], entry["distance"]): entry
                for entry in fixture_file["fixtures"].values()}
    active_world = None
    arena = None
    set_candidate_render_distance(args.output)
    with (args.output / "pose-controls.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        try:
            for seed in seeds[args.worker::args.workers]:
                for distance in args.distances:
                    expected = fixtures[seed, distance]
                    world = (expected["world_seed"], expected["optional_effects"])
                    try:
                        if active_world != world:
                            if arena is not None:
                                arena.sim.close()
                            arena = None
                            active_world = None
                            arena = RevisedCaptureArena(*world)
                            active_world = world
                        for row in run_case(arena, expected, args.output, args.terminal_tolerance):
                            writer.writerow(row)
                            stream.flush()
                    except RuntimeError as error:
                        writer.writerow((distance, seed, expected["world_seed"], "", False, 0,
                                         expected["return_limit"], "", "", "", "", "", "error", "", str(error)))
                        stream.flush()
        finally:
            if arena is not None:
                arena.sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--distances", type=int, nargs="+", choices=DISTANCES, default=(30, 100))
    parser.add_argument("--terminal-tolerance", type=float, default=0.4)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers <= 32 or not 0 < args.terminal_tolerance <= 0.4:
        parser.error("Invalid pose-control shard or terminal tolerance")
    run(args)


if __name__ == "__main__":
    main()
