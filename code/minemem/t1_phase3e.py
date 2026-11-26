"""Build, acceptance-test, and collect the first-arrival T1 v3 benchmark."""

import argparse
import csv
import json
import math
import random
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np

from t1_collect import DART_PROBABILITY, DART_QUOTAS, HELD_OUT_QUOTAS, TRAIN_QUOTAS, write_episode
from t1_dual_enclosure import EnclosureArena, configure_rendering
from t1_pose_replay import return_by_pose
from t1_respawn import record_config
from t1_safety import POSE_FIELDS, StepTrace
from t1_shortcut_audit import evaluate_control, gold_pixels
from t1_structure_audit import find_structure_policy
from t1_v4 import DISTANCES, SUCCESS_RADIUS, bearing, horizontal


PROTOCOL = "t1_v3_first_arrival"
SCENARIO_BASE = 2086000000
WORLD_BASE = 21460000
ACCEPTANCE_SEEDS = tuple(range(SCENARIO_BASE, SCENARIO_BASE + 30))
SPLIT_OFFSETS = {"train": 1, "validation": 2, "test": 3, "dart": 4}
BASELINES = ("teacher", "turn_180", "scan", "find_structure", "reverse_route", "start_pose")
FIELDS = ("distance", "seed", "world_seed", "goal_slot", "baseline", "success", "steps", "return_limit",
          "min_dist", "deaths", "pose_jumps", "end_reason", "initial_gold_distance", "initial_gold_pixels", "error")


def frozen_gold_slots(seeds):
    strata = {(heading, last): [] for heading, last in product(range(4), (0, 1))}
    for index, seed in enumerate(seeds):
        rng = random.Random(seed)
        heading = rng.randrange(4)
        last = rng.sample((0, 1), 2)[-1]
        strata[heading, last].append(index)
    counts = [len(indices) for indices in strata.values()]
    allocations = []
    for left_counts in product(*(range(count//2, (count+1)//2+1) for count in counts)):
        right_counts = [count-left for count, left in zip(counts, left_counts, strict=True)]
        last_matches = sum(left_counts[::2]) + sum(right_counts[1::2])
        if sum(left_counts) == last_matches == len(seeds)//2 and all(
            abs(sum(left_counts[i:i+2])-sum(right_counts[i:i+2])) <= 1 for i in range(0, 8, 2)
        ):
            allocations.append(left_counts)
    if not allocations:
        raise ValueError("Cannot balance gold assignment by arena orientation and visit order")
    rng = random.Random("t1-v3-gold-slots")
    left_counts = rng.choice(allocations)
    left = {index for indices, count in zip(strata.values(), left_counts, strict=True)
            for index in rng.sample(indices, count)}
    return tuple(0 if index in left else 1 for index in range(len(seeds)))


GOLD_ORDER = frozen_gold_slots(ACCEPTANCE_SEEDS)


def collection_gold_slot(seed):
    # Consecutive trials use opposite assignments from a stream independent of layout.
    return (random.Random(f"phase3e-collection-gold:{seed//2}").randrange(2)+seed%2) % 2


class FirstArrivalArena(EnclosureArena):
    door_jamb = 3
    spawn_offset = -17
    distractor = None
    perturb_return = False

    def spawn_heading_target(self, layout):
        return self.project(0, 0)

    def navigation_step(self, action):
        valid = super().navigation_step(action)
        if valid and self.distractor is not None and math.dist(horizontal(self.info), self.distractor) <= SUCCESS_RADIUS:
            self.trace.reason = "distractor_first"
            return False
        return valid

    def executed_return_action(self, correction):
        if not self.perturb_return:
            return correction
        if self.noise_remaining == 0:
            if self.noise.random() >= DART_PROBABILITY:
                return correction
            self.noise_remaining = self.noise.randint(1, 3)
            self.noise_yaw = self.noise.choice((-10.0, -5.0, 0.0, 5.0, 10.0))
        self.noise_remaining -= 1
        action = self.sim.noop_action()
        action["camera"] = np.array([0.0, self.noise_yaw], dtype=np.float32)
        return action

    def setup(self, seed, distance, gold):
        layout = self.layout(seed, distance)
        self.establish(layout, gold)
        distance_from_spawn = math.dist(horizontal(self.info), layout["landmark"])
        if distance_from_spawn < 8:
            raise RuntimeError(f"Spawn is too close to the gold pillar: {distance_from_spawn:.3f}")
        routes = [self.return_targets(layout, slot) for slot in (0, 1)]
        length = sum(math.dist(a, b) for a,b in zip([layout["endpoint"], *routes[gold]], routes[gold]))
        layout.update(protocol=PROTOCOL, teacher_targets=[*routes[gold], layout["landmark"]], candidate_routes=routes,
                      distractor=layout["candidate_positions"][1-gold], spawn_gold_distance=distance_from_spawn,
                      waypoint_conditioned=True, return_limit=round(2*(length/.16+60)+200))
        self.history = deque(maxlen=2)
        self.outbound, self.path_len_out = [], 0.0
        return layout

    def outward(self, layout, trace):
        with self.track(trace):
            observations = []
            dwell_frames = {}
            for room in layout["outbound_room_order"]:
                position = self.project(room, -10)
                _, reached, _ = self.drive(position, .35, 350, record=True)
                if not reached:
                    raise RuntimeError("The teacher could not reach an initial distant observation point")
                target = self.project(room, 0)
                self.face(bearing(horizontal(self.info), target), record=True)
                dwell_frames[room] = len(self.outbound)
                x,z = map(math.floor, target)
                pillar = (x, self.y, z, x, self.y+2, z)
                views = []
                for _ in range(12):
                    pose = tuple(float(self.info["player_pos"][key]) for key in POSE_FIELDS)
                    pixels = gold_pixels(self.info["pov"], None, pose, pillar, layout["occlusion_boxes"])
                    views.append(dict(slot=int(room>0), distance=math.dist(horizontal(self.info), target),
                                      gold_pixels=pixels, pose=pose))
                    self.teacher_step(record=True)
                observations.append(max(views, key=lambda view: view["gold_pixels"]))
            gold, = (item for item in observations if item["slot"] == layout["goal_slot"])
            if gold["gold_pixels"] <= 0 or gold["distance"] < 8:
                raise RuntimeError(f"Distant observation gate failed: {observations}")
            layout.update(initial_observation_pose=gold["pose"], initial_gold_distance=gold["distance"],
                          initial_gold_pixels=gold["gold_pixels"], initial_observation_checks=observations,
                          distant_frame_indices=[dwell_frames[room] for room in sorted(dwell_frames)])
            self.traverse_outbound(layout)
        layout["path_len_out"] = self.path_len_out
        self.landmark = layout["landmark"]
        self.distractor = layout.get("distractor")

    def collect_route(self, layout, dart):
        self.restore_return_pose(layout)
        self.noise = random.Random(f"phase3e-dart:{layout['scenario_seed']}")
        self.noise_remaining = 0
        self.perturb_return = dart
        self.capture = []
        reached = False
        trace = StepTrace(None)
        try:
            with self.track(trace):
                used = 0
                for index, target in enumerate(layout["teacher_targets"]):
                    final = index == len(layout["teacher_targets"])-1
                    count, reached, _ = self.drive(target, SUCCESS_RADIUS if final else .40,
                                                    layout["return_limit"]-used)
                    used += count
                    if not reached:
                        break
            if not reached or trace.reason:
                raise RuntimeError(f"Teacher return failed: {trace.reason or 'budget'}")
            layout["teacher_return_steps"] = len(self.capture)
            return self.capture
        finally:
            self.capture = None
            self.perturb_return = False
            self.distractor = None


def turn_around(frame, move):
    for _ in range(18):
        frame = move(yaw=10)
    while True:
        frame = move(forward=True)


def reverse_waypoints(poses):
    # Retain places where the agent paused. Pure distance subsampling can omit a visit at a U-turn.
    paused = set()
    stable = 0
    for index, (before,after) in enumerate(zip(poses, poses[1:]), start=1):
        stable = stable+1 if math.dist(before, after) < .005 else 0
        if stable == 6:
            paused.add(index)
    route = [poses[-1]]
    for index in range(len(poses)-2, -1, -1):
        separation = math.dist(poses[index], route[-1])
        if separation >= .85 or (index in paused and separation >= .20):
            route.append(poses[index])
    if route[-1] != poses[0]:
        route.append(poses[0])
    return route[1:]


def acceptance_outbound(arena, seed, distance, gold, output):
    tag = f"d{distance}-seed{seed}"
    layout = arena.setup(seed, distance, gold)
    outbound = StepTrace(output / f"{tag}-outbound.csv.gz")
    try:
        arena.outward(layout, outbound)
    finally:
        arena.distractor = None
    poses = [(row[1], row[3]) for row in outbound.rows]
    return layout, poses


def acceptance_case(arena, layout, poses, output):
    seed, distance, gold = (layout[key] for key in ("scenario_seed", "distance", "goal_slot"))
    tag = f"d{distance}-seed{seed}"
    scene = arena.setup(seed, distance, gold)
    for key in ("walls", "candidate_positions", "spawn", "return_pose", "teacher_targets"):
        if scene[key] != layout[key]:
            raise RuntimeError(f"Rebuilt phase 3e geometry changed {key}")
    arena.landmark, arena.distractor = layout["landmark"], layout["distractor"]
    start, reverse = poses[0], reverse_waypoints(poses)
    for name in BASELINES:
        trace = StepTrace(output / f"{tag}-{name}.csv.gz")
        if name in ("teacher", "start_pose", "reverse_route"):
            targets = {"teacher": layout["teacher_targets"], "start_pose": [start], "reverse_route": reverse}
            ok, steps, _, close, _, deaths, _, jumps, reason = return_by_pose(
                arena, layout, targets[name], trace, terminal_tolerance=.10,
                waypoint_tolerance=.25 if name == "reverse_route" else .40)
            score = dict(success=ok, steps=steps, min_dist=close, deaths=deaths, pose_jumps=jumps, end_reason=reason)
            if name == "teacher":
                layout.update(teacher_return_steps=steps, teacher_success=ok)
                if ok:
                    layout["return_limit"] = 2*steps+200
                record_config(output, tag, layout)
        else:
            policy = {"turn_180": turn_around, "find_structure": find_structure_policy, "scan": None}[name]
            score = evaluate_control(arena, layout, trace, name == "scan", policy=policy)
        yield (distance, seed, arena.world_seed, gold, name, score["success"], score["steps"],
               layout["return_limit"], score["min_dist"], score["deaths"], score["pose_jumps"],
               score["end_reason"], layout["initial_gold_distance"], layout["initial_gold_pixels"], "")
    arena.distractor = None


def acceptance(args):
    with (args.output / "acceptance.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for index in range(args.seeds)[args.worker::args.workers]:
            seed = ACCEPTANCE_SEEDS[index]
            arena = FirstArrivalArena(WORLD_BASE+index, effects=True)
            try:
                staged = []
                for distance in args.distances:
                    try:
                        staged.append(acceptance_outbound(arena, seed, distance, GOLD_ORDER[index], args.output))
                    except RuntimeError as error:
                        writer.writerow((distance, seed, arena.world_seed, GOLD_ORDER[index], *([""]*10), str(error)))
                        stream.flush()
                for layout, poses in staged:
                    try:
                        for row in acceptance_case(arena, layout, poses, args.output):
                            writer.writerow(row)
                            stream.flush()
                    except RuntimeError as error:
                        writer.writerow((layout["distance"], seed, arena.world_seed, GOLD_ORDER[index],
                                         *([""]*10), str(error)))
                        stream.flush()
                    finally:
                        arena.distractor = None
            finally:
                arena.sim.close()


def capture_case(arena, seed, group, dart, output, split):
    if group == "visible":
        arena.distractor = None
        arena.landmark = arena.x+.5, arena.z+.5
        layout = arena.prepare_visible(seed)
        layout.update(teacher_targets=[layout["landmark"]], waypoint_conditioned=True, protocol=PROTOCOL,
                      return_limit=200)
    else:
        layout = arena.setup(seed, group, collection_gold_slot(seed))
    if dart:
        layout["dart"] = {"probability": DART_PROBABILITY, "duration": [1, 3]}
    arena.capture = []
    try:
        if group == "visible":
            for _ in range(2):
                arena.teacher_step()
        else:
            arena.outward(layout, StepTrace(None))
        outbound = arena.capture
    finally:
        arena.capture = None
    returns = arena.collect_route(layout, dart)
    return write_episode(output, layout, outbound, returns, split, group)


def collect(args):
    offset = SPLIT_OFFSETS[args.split]
    split = "train" if args.split == "dart" else args.split
    quotas = DART_QUOTAS if args.split == "dart" else (TRAIN_QUOTAS if split == "train" else HELD_OUT_QUOTAS)
    arena = FirstArrivalArena(WORLD_BASE+offset*1000+args.worker, effects=True)
    try:
        with (args.output / "episodes.jsonl").open("x") as stream:
            for g, (group, quota) in enumerate(quotas.items()):
                needed = math.ceil(quota*args.fraction/args.workers)
                frames = 0
                for trial in range(100):
                    if frames >= needed:
                        break
                    seed = SCENARIO_BASE+offset*10000000+args.worker*10000+g*1000+trial
                    try:
                        row = capture_case(arena, seed, group, args.split == "dart", args.output, split)
                    except RuntimeError as error:
                        arena.distractor = None
                        print(f"retry seed={seed} group={group}: {error}", flush=True)
                        continue
                    stream.write(json.dumps(row)+"\n")
                    stream.flush()
                    frames += row["frames"]
                if frames < needed:
                    raise RuntimeError(f"Collection quota failed: group={group} frames={frames} expected={needed}")
    finally:
        arena.sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--distances", type=int, nargs="+", choices=DISTANCES, default=DISTANCES)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--split", choices=tuple(SPLIT_OFFSETS))
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers <= 32 or not 0 < args.seeds <= 30 or not 0 < args.fraction <= 1:
        parser.error("Invalid CPU shard, seed count or collection fraction")
    configure_rendering(args.output)
    collect(args) if args.split else acceptance(args)
