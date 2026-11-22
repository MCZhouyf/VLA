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
from t1_respawn import freeze_return, record_config, verify_scene
from t1_respawn_report import acceptance_summary
from t1_safety import SETUP_REVISION, ProtectedArena, StepTrace, TrialInterrupted
from t1_v4 import (
    DISTANCES, INSTRUCTION, SUCCESS_RADIUS, VISIBLE_INSTRUCTION, horizontal,
)


NAV_SEED = 2030990000
NAV_WORLD_SEED = 19961103


class ProtectedCaptureArena(ProtectedArena, CaptureArena):
    def visible_trajectory(self, seed, exact_distance=None):
        self.run_commands(self.protection_commands)
        config, outbound, returns = super().visible_trajectory(seed, exact_distance)
        config.update(self.safety_config())
        return config, outbound, returns


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


def run_case(arena, controller, config, outbound, instruction, limit, trace):
    arena.restore_return_pose(config)
    controller.start(outbound)
    closest = math.dist(horizontal(arena.info), arena.landmark)
    tags = 0
    final = closest
    steps = 0
    start = perf_counter()
    try:
        with arena.track(trace):
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
    except TrialInterrupted:
        pass
    ok = closest <= SUCCESS_RADIUS and not trace.reason
    return (ok, steps, final, closest, tags, perf_counter() - start,
            *trace.flags(), trace.reason or ("reached" if ok else "budget"))


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
                       or row.get("setup_revision") != SETUP_REVISION
                       or row["success"] != str(float(row["min_dist"]) <= SUCCESS_RADIUS
                                                and int(row["deaths"]) == 0
                                                and row["pose_jump"] == "False")
                       for row in nav)
                or sum(row["success"] == "True" for row in nav) < 8):
            raise SystemExit("The no-memory model did not pass visible navigation at 8/10")
        gate_runs = {row["training_run"] for row in nav}
        if len(gate_runs) != 1 or (settings["variant"] == "no_memory" and gate_runs != {training_run}):
            raise SystemExit("The gate must come from the no-memory run being evaluated")
        if not all(row["accepted"] for row in acceptance_summary(args.acceptance)):
            raise SystemExit("T1 respawn teacher and shortcut gate did not pass")
        frozen = json.loads(args.seeds_file.read_text())
        if args.seed not in frozen["formal"]:
            raise ValueError("Select one frozen formal scenario seed")
        fixtures = fixture_index(args.acceptance)
        world = frozen["world_seed_by_scenario"][str(args.seed)]
        effects = fixtures[DISTANCES[0], args.seed]["optional_effects"]
    else:
        world = NAV_WORLD_SEED
        effects = False
    controller = Controller(args.checkpoint, args.training_run)
    variant = controller.settings["variant"]
    arena = ProtectedCaptureArena(world, effects)
    try:
        with (args.output / "episodes.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("mode", "model", "seed", "d", "world_seed", "success", "steps",
                             "final_dist", "min_dist", "decoded_steps", "elapsed_seconds",
                             "deaths", "pose_jump", "pose_jumps", "end_reason", "return_limit",
                             "path_len_out", "training_run", "setup_revision"))
            if args.mode in ("nav", "smoke"):
                count = 10 if args.mode == "nav" else 1
                limit = 200 if args.mode == "nav" else 30
                for seed in range(NAV_SEED, NAV_SEED + count):
                    config, outbound, _ = arena.visible_trajectory(seed, exact_distance=5)
                    config["return_limit"] = limit
                    key = f"d5-seed{seed}"
                    record_config(args.output, key, config)
                    trace = StepTrace(args.output / f"{key}-return.csv.gz")
                    result = run_case(arena, controller, config, outbound, VISIBLE_INSTRUCTION, limit, trace)
                    writer.writerow((args.mode, variant, seed, 5, world, *result, limit, "",
                                     training_run, SETUP_REVISION))
                    stream.flush()
            else:
                for distance in DISTANCES:
                    expected = fixtures[distance, args.seed]
                    config = arena.prepare_episode(args.seed, distance)
                    verify_scene(config, expected)
                    key = f"d{distance}-seed{args.seed}"
                    outbound_trace = StepTrace(args.output / f"{key}-outbound.csv.gz")
                    arena.capture = []
                    try:
                        with arena.track(outbound_trace):
                            arena.outbound_segment(config)
                        outbound = arena.capture
                    except TrialInterrupted:
                        result = (False, 0, distance, distance, 0, 0, *outbound_trace.flags(),
                                  f"outbound_{outbound_trace.reason}")
                        config["path_len_out"] = arena.path_len_out
                    finally:
                        arena.capture = None
                    if not outbound_trace.reason:
                        freeze_return(config, expected)
                        arena.command("/gamemode adventure @p")
                        arena.settle()
                        trace = StepTrace(args.output / f"{key}-return.csv.gz")
                        result = run_case(arena, controller, config, outbound, INSTRUCTION,
                                          expected["return_limit"], trace)
                        arena.command("/gamemode survival @p")
                    record_config(args.output, key, config)
                    writer.writerow((args.mode, variant, args.seed, distance, world, *result,
                                     expected["return_limit"], config["path_len_out"], training_run,
                                     SETUP_REVISION))
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
