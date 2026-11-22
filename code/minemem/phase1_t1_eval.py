"""Evaluate T1 baselines with a shared teacher outbound segment and frozen limits."""

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from phase1_metrics import gate, teacher
from phase1_t1 import Arena, DISTANCES, SUCCESS_RADIUS, horizontal
from phase1_t1_report import validate_results


INSTRUCTION = "Go back to the gold pillar you saw."
FIELDS = ("model", "d", "seed", "success", "steps", "final_dist", "min_dist",
          "path_len_out", "world_seed", "return_limit", "elapsed_seconds")


class Jarvis:
    def __init__(self, work, base_url):
        from jarvisvla.evaluate.agent_wrapper import VLLM_AGENT

        self.agent = VLLM_AGENT(
            checkpoint_path=str(work / "models/JarvisVLA-Qwen2-VL-7B"), base_url=base_url,
            history_num=2, action_chunk_len=1, instruction_type="normal", temperature=0.9,
        )

    def start(self, arena, seed):
        self.agent.reset()
        self.agent.actions.clear()
        self.agent.history = [
            (self.agent.processor_wrapper.create_image_input(frame), action, "", index)
            for index, (frame, action) in enumerate(arena.history)
        ]
        if len(self.agent.history) != self.agent.history_num:
            raise RuntimeError("The teacher segment did not provide the configured history")

    def action(self, arena):
        controls = self.agent.forward([arena.info["pov"]], [INSTRUCTION])
        return arena.sim.agent_action_to_env_action(controls)


class MineStudioPolicy:
    def __init__(self, work, name, device):
        import torch
        from minestudio.simulator.entry import CameraConfig
        from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
        from minestudio.utils.vpt_lib.actions import ActionTransformer

        if name == "steve1":
            from minestudio.models.steve_one import SteveOnePolicy
            model = SteveOnePolicy
            checkpoint = "MineStudio_STEVE-1.official"
        else:
            from minestudio.models.vpt import VPTPolicy
            model = VPTPolicy
            checkpoint = "MineStudio_VPT.bc_early_game_3x"
        if device == "cuda" and torch.cuda.device_count() != 1:
            raise RuntimeError("A lightweight GPU policy requires a single allocated CUDA device")
        self.policy = model.from_pretrained(work / "models" / checkpoint).to(device).eval()
        self.device = device
        self.name = name
        camera = CameraConfig()
        self.mapper = CameraHierarchicalMapping(n_camera_bins=camera.n_camera_bins)
        self.transformer = ActionTransformer(**camera.action_transformer_kwargs)
        self.torch = torch

    def start(self, arena, seed):
        self.torch.manual_seed(seed)
        self.state = None
        with self.torch.inference_mode():
            self.condition = (
                self.policy.prepare_condition({"text": INSTRUCTION, "cond_scale": 4.0})
                if self.name == "steve1" else None
            )

    def action(self, arena):
        import cv2

        image = cv2.resize(arena.info["pov"], (128, 128), interpolation=cv2.INTER_LINEAR)
        inputs = {"image": self.torch.from_numpy(image).to(self.device)[None, None]}
        if self.condition is not None:
            inputs["condition"] = self.condition
        with self.torch.inference_mode():
            action, self.state = self.policy.get_action(inputs, state_in=self.state)
        indices = {key: np.asarray(value.item()) for key, value in action.items()}
        return self.transformer.policy2env(self.mapper.to_factored(indices))


class RandomPolicy:
    def start(self, arena, seed):
        self.space = arena.sim.action_space
        self.space.seed(seed)

    def action(self, arena):
        return self.space.sample()


def return_segment(arena, policy, config, seed, steps_limit, navigation_step=None):
    arena.restore_return_pose(config)
    policy.start(arena, seed)
    started = perf_counter()
    final_dist = mindist = math.dist(horizontal(arena.info), arena.landmark)
    for step in range(1, steps_limit + 1):
        action = policy.action(arena)
        if navigation_step is None:
            _, _, terminated, truncated, arena.info = arena.sim.step(action)
            done = terminated or truncated
        else:
            if not navigation_step(action):
                break
            done = False
        final_dist = math.dist(horizontal(arena.info), arena.landmark)
        mindist = min(mindist, final_dist)
        if final_dist <= SUCCESS_RADIUS or done:
            break
    return mindist <= SUCCESS_RADIUS, step, final_dist, mindist, perf_counter() - started


def fixture_index(directories):
    fixtures = {}
    for directory in directories:
        for path in sorted(directory.glob("worker-*/d*-seed*.json")):
            config = json.loads(path.read_text())
            key = config["distance"], config["scenario_seed"]
            if key in fixtures:
                raise ValueError(f"Duplicate teacher fixture {key}")
            fixtures[key] = config
    return fixtures


def make_policies(args):
    return {
        "random": RandomPolicy(),
        "steve1": MineStudioPolicy(args.work, "steve1", args.lite_device),
        "vpt": MineStudioPolicy(args.work, "vpt", args.lite_device),
        "jarvis7b": Jarvis(args.work, args.base_url),
    }


def evaluate_fixture(arena, fixture, policies, writer, navigation_step=None):
    d, seed = fixture["distance"], fixture["scenario_seed"]
    config = arena.prepare_episode(seed, d)
    for key in ("landmark", "arena_origin", "wall", "waypoints", "planned_endpoint"):
        if json.loads(json.dumps(config[key])) != fixture[key]:
            raise RuntimeError(f"Teacher fixture changed at d={d}, seed={seed}: {key}")
    arena.outbound_segment(config)
    if abs(config["actual_distance"] - fixture["actual_distance"]) > 0.65:
        raise RuntimeError(f"Teacher endpoint differs from frozen fixture at d={d}, seed={seed}")
    # Preserve the same blocks across policies, including implementations that
    # emit attack or use. The benchmark is distance-based navigation.
    arena.command("/gamemode adventure @p")
    arena.settle()
    for model, policy in policies.items():
        ok, steps, final, closest, seconds = return_segment(
            arena, policy, config, seed, fixture["return_limit"], navigation_step=navigation_step)
        writer.writerow(dict(zip(FIELDS, (model, d, seed, ok, steps, final, closest,
                                          config["path_len_out"], arena.world_seed,
                                          fixture["return_limit"], seconds), strict=True)))
    arena.command("/gamemode survival @p")


def evaluate(args):
    if not all(result["accepted"] for result in gate(args.gate_results)):
        raise SystemExit("JARVIS-VLA gate has not passed")
    teacher_inputs = [path for directory in args.teacher_dir
                      for path in sorted(directory.glob("worker-*/teacher.csv"))]
    if not all(result["accepted"] for result in teacher(teacher_inputs, 50)):
        raise SystemExit("T1 teacher acceptance has not passed")
    fixtures = fixture_index(args.teacher_dir)
    valid = sorted(set.intersection(*(
        {seed for (distance, seed), config in fixtures.items() if distance == d
         and "return_limit" in config}
        for d in DISTANCES
    )))
    if len(valid) < 35:
        raise RuntimeError("The frozen teacher evidence does not contain 35 common valid seeds")
    frozen = json.loads(args.seeds_file.read_text())
    expected = dict(protocol="t1_v3", pilot=valid[:5], formal=valid[5:35],
                    world_seed_by_scenario={str(seed): fixtures[10, seed]["world_seed"]
                                            for seed in valid})
    if frozen != expected:
        raise RuntimeError("The frozen T1 seed list does not match the accepted teacher fixtures")
    pilot, formal = frozen["pilot"], frozen["formal"]
    if args.end == 30:
        if args.pilot_dir is None:
            raise SystemExit("Provide the completed T1 pilot before the formal run")
        validate_results(sorted(args.pilot_dir.glob("*/cliff_T1.csv")), pilot)
    stage_seeds = pilot if args.end == 5 else formal
    cases = stage_seeds[args.start + args.shard:args.end:args.shards]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "seeds.json").open("x") as output:
        json.dump(dict(pilot=pilot, formal=formal, worker=cases), output, indent=2)
        output.write("\n")
    policies = make_policies(args)
    with (args.output / "cliff_T1.csv").open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        output.flush()
        for seed in cases:
            arena = Arena(fixtures[10, seed]["world_seed"])
            try:
                for d in DISTANCES:
                    fixture = fixtures[d, seed]
                    if fixture["world_seed"] != arena.world_seed or "return_limit" not in fixture:
                        raise RuntimeError(f"Missing matching successful teacher fixture {d}, {seed}")
                    evaluate_fixture(arena, fixture, policies, writer)
                    output.flush()
            finally:
                arena.sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gate-results", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds-file", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--lite-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, choices=(5, 30), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.start < args.end or not 0 <= args.shard < args.shards:
        parser.error("Invalid seed range or shard")
    evaluate(args)


if __name__ == "__main__":
    main()
