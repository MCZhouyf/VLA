"""Collect balanced T1 and visible-target teacher trajectories on CPU."""

import argparse
import json
import math
import random
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from t1_v4 import (
    DISTANCES, INSTRUCTION, PROTOCOL, SUCCESS_RADIUS, VISIBLE_INSTRUCTION,
    RandomHeadingArena, bearing, horizontal,
)


SPLITS = ("train", "validation", "test")
SCENARIO_BASE = 2030010000
WORLD_BASE = 20710100
TRAIN_QUOTAS = {**dict.fromkeys(DISTANCES, 25000), "visible": 20000}
HELD_OUT_QUOTAS = {**dict.fromkeys(DISTANCES, 1500), "visible": 500}


class CaptureArena(RandomHeadingArena):
    def __init__(self, world_seed):
        self.capture = None
        self.candidates = []
        super().__init__(world_seed)

    def teacher_step(self, yaw_delta=0.0, forward=False, jump=False, record=False):
        if self.capture is None:
            return super().teacher_step(yaw_delta, forward, jump, record)

        import cv2

        pose = self.info["player_pos"]
        frame = cv2.cvtColor(self.info["pov"], cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            raise RuntimeError("Could not encode teacher observation")
        action = self.sim.noop_action()
        action.update(camera=np.array([0.0, yaw_delta], dtype=np.float32),
                      forward=int(forward), jump=int(jump))
        if record:
            super().teacher_step(yaw_delta, forward, jump, record)
            token = self.outbound[-1][-1]
        else:
            batch = {key: np.asarray([value]) for key, value in action.items()}
            agent = self.sim.env_action_to_agent_action(batch)
            token = self.tokenizer.encode_action(
                (int(agent["buttons"].item()), int(agent["camera"].item()))
            )
            self.tick(action)
        self.capture.append((encoded.tobytes(), [float(pose[key]) for key in ("x", "z", "yaw")],
                             token))

    def follow_return(self, targets, distance):
        self.capture = []
        try:
            metrics = super().follow_return(targets, distance)
            self.candidates.append(self.capture)
            return metrics
        finally:
            self.capture = None

    def t1_trajectory(self, seed, distance):
        config = self.prepare_episode(seed, distance)
        self.capture = []
        try:
            self.outbound_segment(config)
            outbound = self.capture
        finally:
            self.capture = None
        self.candidates = []
        success, _, _, _ = self.teacher_return(config)
        if not success:
            raise RuntimeError("Teacher did not reach the T1 landmark")
        index = min(range(len(self.candidates)),
                    key=lambda i: (not config["teacher_route_attempts"][i]["success"],
                                   config["teacher_route_attempts"][i]["steps"],
                                   config["teacher_route_attempts"][i]["final_dist"]))
        return config, outbound, self.candidates[index]

    def visible_trajectory(self, seed, exact_distance=None):
        rng = random.Random(seed)
        distance = exact_distance if exact_distance is not None else rng.randrange(5, 11)
        for box in self.scene_fills:
            self.fill(*box, "air")
        self.scene_fills.clear()
        gold = self.scene_box(self.landmark, self.landmark, "gold_block")
        angle = rng.uniform(-math.pi, math.pi)
        start = (self.landmark[0] + distance * math.sin(angle),
                 self.landmark[1] + distance * math.cos(angle))
        yaw = bearing(start, self.landmark)
        self.settle(8)
        self.command(f"/tp @p {start[0]:.6f} {self.y} {start[1]:.6f} {yaw:.6f} 0")
        self.settle()
        self.check_blocks([(gold[0], self.y + 1, gold[2], "gold_block")])
        if math.dist(horizontal(self.info), start) > 0.15:
            raise RuntimeError("Could not establish the visible-target start pose")
        config = dict(protocol=PROTOCOL, scenario_seed=seed, world_seed=self.world_seed,
                      distance=distance, landmark=self.landmark,
                      return_pose=self.info["player_pos"], visible=True)
        self.capture = []
        try:
            for _ in range(2):
                self.teacher_step()
            outbound = self.capture
            self.capture = []
            count, reached, closest = self.drive(self.landmark, SUCCESS_RADIUS, 200)
            if not reached:
                raise RuntimeError(f"Visible-target teacher did not arrive, closest={closest}")
            config.update(teacher_return_steps=count, return_limit=200)
            return config, outbound, self.capture
        finally:
            self.capture = None


def write_episode(output, config, outbound, returns, split, group):
    frames = outbound + returns
    name = f"episode-{config['scenario_seed']}.zip"
    path = output / name
    partial = output / (name + ".partial")
    if path.exists():
        raise FileExistsError(path)
    metadata = {
        "scene": config, "split": split, "group": group,
        "instruction": VISIBLE_INSTRUCTION if group == "visible" else INSTRUCTION,
        "outbound_frames": len(outbound),
        "pose": [frame[1] for frame in frames],
        "action": [frame[2] for frame in frames],
    }
    with ZipFile(partial, "x", compression=ZIP_STORED) as archive:
        archive.writestr("episode.json", json.dumps(metadata, default=float))
        for index, (image, _, _) in enumerate(frames):
            archive.writestr(f"{index:05d}.jpg", image)
    partial.replace(path)
    return {"path": name, "scene_seed": config["scenario_seed"], "world_seed": config["world_seed"],
            "split": split, "group": group, "frames": len(frames), "outbound_frames": len(outbound),
            "return_frames": len(returns)}


def collect(args):
    split_index = SPLITS.index(args.split)
    quotas = TRAIN_QUOTAS if args.split == "train" else HELD_OUT_QUOTAS
    targets = {
        group: math.ceil(quota * args.fraction / args.workers)
        for group, quota in quotas.items()
    }
    done = dict.fromkeys(targets, 0)
    world = WORLD_BASE + 1000 * split_index + args.worker
    start_seed = SCENARIO_BASE + 100000 * split_index + 1000 * args.worker
    counter = 0
    for attempt in range(4):
        try:
            arena = CaptureArena(world + 100 * attempt)
            break
        except RuntimeError as error:
            print(f"World {world + 100 * attempt} setup failed: {error}", flush=True)
            if attempt == 3:
                raise
    try:
        with (args.output / "episodes.jsonl").open("x") as index:
            while any(done[group] < targets[group] for group in targets):
                for group, quota in targets.items():
                    if done[group] >= quota:
                        continue
                    if counter >= 1000:
                        raise RuntimeError("Exhausted the worker's disjoint scenario seed range")
                    seed = start_seed + counter
                    counter += 1
                    try:
                        if group == "visible":
                            trajectory = arena.visible_trajectory(seed)
                        else:
                            trajectory = arena.t1_trajectory(seed, group)
                    except RuntimeError as error:
                        print(f"Rejected {seed} in {group}: {error}", flush=True)
                        continue
                    record = write_episode(args.output, *trajectory, args.split, group)
                    done[group] += record["frames"]
                    index.write(json.dumps(record) + "\n")
                    index.flush()
                    print(json.dumps(record), flush=True)
    finally:
        arena.sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers <= 32 or not 0 < args.fraction <= 1:
        parser.error("Invalid worker shard or collection fraction")
    collect(args)


if __name__ == "__main__":
    main()
