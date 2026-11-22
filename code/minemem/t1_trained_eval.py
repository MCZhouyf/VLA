"""Evaluate trained T1 policies in MineStudio using the frozen protocol."""

import argparse
import csv
import io
import json
import math
from collections import deque
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

from phase1_t1_eval import fixture_index
from t1_collect import CaptureArena
from t1_model import T1Policy
from t1_v4 import (
    DISTANCES, INSTRUCTION, PROTOCOL, SUCCESS_RADIUS, VISIBLE_INSTRUCTION,
    horizontal, wrapped_angle,
)
from t1_v4_report import summarize


NAV_SEED = 2030990000
NAV_WORLD_SEED = 19961103


class Controller:
    def __init__(self, checkpoint, training_run):
        self.settings = json.loads((training_run / "training_config.json").read_text())
        self.policy = T1Policy(checkpoint, self.settings["variant"],
                               quantized=self.settings["quantized"], adapter=training_run / "adapter")
        if self.policy.memory is not None:
            from safetensors.torch import load_file

            self.policy.memory.load_state_dict(load_file(training_run / "spatial.safetensors"))
        self.policy.eval()

    def image(self, frame):
        resolution = self.settings.get("resolution")
        if resolution:
            frame = frame.resize(tuple(resolution), Image.Resampling.BILINEAR)
        return self.policy.images.create_image_input(frame)

    def start(self, outbound):
        images = []
        actions = []
        poses = []
        if self.policy.memory is None:
            outbound = outbound[-self.policy.history:]
        for jpeg, pose, action in outbound:
            with Image.open(io.BytesIO(jpeg)) as frame:
                images.append(self.image(frame.convert("RGB")))
            poses.append(pose)
            actions.append(action)
        self.history = deque(zip(images, actions), maxlen=self.policy.history)
        self.poses = poses
        self.features = []
        if self.policy.memory is not None:
            for begin in range(0, len(images), 8):
                self.features.extend(self.policy.pooled_images(images[begin:begin + 8]))

    def action(self, info, instruction):
        image = self.image(Image.fromarray(info["pov"]))
        pose = [float(info["player_pos"][key]) for key in ("x", "z", "yaw")]
        frames = [frame for frame, _ in self.history] + [image]
        actions = [action for _, action in self.history]
        if len(actions) < self.policy.history:
            missing = self.policy.history - len(actions)
            frames = [frames[0]] * missing + frames
            actions = [self.policy.actions.null_token()] * missing + actions
        features = torch.stack(self.features) if self.features else torch.empty(
            (0, self.policy.base.config.hidden_size))
        control, generated, tags = self.policy.predict(
            instruction, frames, actions, features, [*self.poses, pose])
        self.history.append((image, generated))
        self.poses.append(pose)
        if self.policy.memory is not None:
            self.features.extend(self.policy.pooled_images([image]))
        return control, tags


def run_case(arena, controller, config, outbound, instruction, limit):
    arena.restore_return_pose(config)
    controller.start(outbound)
    closest = math.dist(horizontal(arena.info), arena.landmark)
    tags = 0
    final = closest
    start = perf_counter()
    for steps in range(1, limit + 1):
        action, complete = controller.action(arena.info, instruction)
        tags += complete
        action = arena.sim.agent_action_to_env_action(action)
        if not arena.navigation_step(action):
            break
        final = math.dist(horizontal(arena.info), arena.landmark)
        closest = min(closest, final)
        if closest <= SUCCESS_RADIUS:
            break
    return closest <= SUCCESS_RADIUS, steps, final, closest, tags, perf_counter() - start


def run(args):
    training_run = str(args.training_run.resolve())
    settings = json.loads((args.training_run / "training_config.json").read_text())
    if args.mode == "formal":
        if not args.no_memory_gate or not args.acceptance or not args.seeds_file:
            raise SystemExit("Formal evaluation requires the passed no-memory navigation gate and T1 fixtures")
        with args.no_memory_gate.open(newline="") as stream:
            nav = list(csv.DictReader(stream))
        if ({int(row["seed"]) for row in nav} != set(range(NAV_SEED, NAV_SEED + 10))
                or len(nav) != 10
                or any(row["model"] != "no_memory" or row["mode"] != "nav"
                       or int(row["d"]) != 5
                       or row["success"] != str(float(row["min_dist"]) <= SUCCESS_RADIUS)
                       for row in nav)
                or sum(row["success"] == "True" for row in nav) < 8):
            raise SystemExit("The no-memory model did not pass visible navigation at 8/10")
        gate_runs = {row["training_run"] for row in nav}
        if len(gate_runs) != 1 or (settings["variant"] == "no_memory" and gate_runs != {training_run}):
            raise SystemExit("The gate must come from the no-memory run being evaluated")
        if not all(row["accepted"] for row in summarize(args.acceptance)):
            raise SystemExit("T1 v4 teacher and shortcut gate did not pass")
        frozen = json.loads(args.seeds_file.read_text())
        if args.seed not in frozen["formal"]:
            raise ValueError("Select one frozen formal scenario seed")
        fixtures = fixture_index(args.acceptance)
        world = frozen["world_seed_by_scenario"][str(args.seed)]
    else:
        world = NAV_WORLD_SEED
    controller = Controller(args.checkpoint, args.training_run)
    variant = controller.settings["variant"]
    arena = CaptureArena(world)
    try:
        with (args.output / "episodes.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("mode", "model", "seed", "d", "world_seed", "success", "steps",
                             "final_dist", "min_dist", "decoded_steps", "elapsed_seconds",
                             "return_limit", "path_len_out", "training_run"))
            if args.mode in ("nav", "smoke"):
                count = 10 if args.mode == "nav" else 1
                limit = 200 if args.mode == "nav" else 30
                for seed in range(NAV_SEED, NAV_SEED + count):
                    config, outbound, _ = arena.visible_trajectory(seed, exact_distance=5)
                    result = run_case(arena, controller, config, outbound, VISIBLE_INSTRUCTION, limit)
                    writer.writerow((args.mode, variant, seed, 5, world, *result, limit, "", training_run))
                    stream.flush()
            else:
                for distance in DISTANCES:
                    expected = fixtures[distance, args.seed]
                    config = arena.prepare_episode(args.seed, distance)
                    if expected["protocol"] != PROTOCOL or list(config["landmark"]) != expected["landmark"]:
                        raise RuntimeError("Formal scene differs from accepted T1 v4 fixture")
                    arena.capture = []
                    arena.outbound_segment(config)
                    outbound = arena.capture
                    arena.capture = None
                    if abs(wrapped_angle(config["final_yaw"] - expected["final_yaw"])) > 2:
                        raise RuntimeError("Formal replay differs from frozen randomized orientation")
                    config["return_pose"] = expected["return_pose"]
                    arena.command("/gamemode adventure @p")
                    arena.settle()
                    result = run_case(arena, controller, config, outbound, INSTRUCTION,
                                      expected["return_limit"])
                    arena.command("/gamemode survival @p")
                    writer.writerow((args.mode, variant, args.seed, distance, world, *result,
                                     expected["return_limit"], config["path_len_out"], training_run))
                    stream.flush()
    finally:
        arena.sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "nav", "formal"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-memory-gate", type=Path)
    parser.add_argument("--acceptance", nargs="+", type=Path)
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--seed", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
