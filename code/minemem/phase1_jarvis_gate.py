"""Run published JARVIS-VLA rollout presets or the five-block navigation check."""

import argparse
import csv
import importlib.util
import inspect
import math
import random
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np


# The released rollout scripts use one action per query, unlike the paper's
# two-action inference description. Keep each published preset as released.
PRESETS = {
    "craft/craft_crafting_table": (400, 0.9, 4, "recipe"),
    "smelt/iron_ingot": (400, 0.9, 4, "recipe"),
    "mine/mine_stone": (500, 0.6, 2, "normal"),
    "kill/kill_zombie": (300, 0.9, 2, "normal"),
}
NAVIGATION = "navigation"
NAVIGATION_INSTRUCTION = "Go to the gold block."
NAVIGATION_LIMIT = 200
NAVIGATION_RADIUS = 1.5


def official_episode(args, seed):
    import hydra
    if importlib.util.find_spec("minestudio.models.shell") is None:
        # The released helpers still import each other under the former
        # MineStudio paths. Use the helpers at the same JARVIS revision.
        from jarvisvla.evaluate.env_helper import gui_agent
        sys.modules.setdefault("minestudio.models.shell.gui_agent", gui_agent)
        from jarvisvla.evaluate.env_helper import craft_agent
        sys.modules.setdefault("minestudio.models.shell.craft_agent", craft_agent)
    from jarvisvla.evaluate import evaluate
    from minestudio.simulator.callbacks.callback import MinecraftCallback

    compose = hydra.compose
    command_callback = evaluate.CommandsCallback
    inventory_callback = evaluate.InitInventoryCallback
    minecraft_sim = evaluate.MinecraftSim
    opened = []

    def compose_with_seed(*positional, **keywords):
        cfg = compose(*positional, **keywords)
        cfg.seed = seed
        return cfg

    def create_sim(*positional, **keywords):
        sim = minecraft_sim(*positional, **keywords)
        opened.append(sim)
        return sim

    def compatible_inventory(inventory, inventory_distraction_level, equip_distraction_level):
        if "equip_distraction_level" in inspect.signature(inventory_callback).parameters:
            return inventory_callback(inventory, inventory_distraction_level=inventory_distraction_level,
                                      equip_distraction_level=equip_distraction_level)
        print("MineStudio inventory-only callback, no equipment distraction", flush=True)
        return inventory_callback(inventory, distraction_level=inventory_distraction_level)

    frames, temperature, history, instruction_type = PRESETS[args.task]
    try:
        with (
            patch.object(hydra, "compose", compose_with_seed),
            patch.object(evaluate, "MinecraftSim", create_sim),
            patch.object(evaluate, "RecordCallback", lambda **_: MinecraftCallback()),
            # Empty command lists are a no-op. This MineStudio revision tries to
            # wrap an already wrapped observation if the empty callback is used.
            patch.object(evaluate, "CommandsCallback",
                         lambda commands: command_callback(commands) if commands else MinecraftCallback()),
            patch.object(evaluate, "InitInventoryCallback", compatible_inventory),
        ):
            success, index = evaluate.evaluate(
                video_path=str(args.output / "unused.mp4"),
                checkpoints=str(args.checkpoint),
                environment_config={"env_config": args.task, "max_frames": frames, "verbos": False},
                model_config={"temperature": temperature, "history_num": history,
                              "instruction_type": instruction_type, "action_chunk_len": 1},
                base_url=args.base_url,
            )
    except BaseException:
        for sim in opened:
            sim.close()
        raise
    steps = index + 1 if success else index
    if steps < 0:
        raise RuntimeError(f"Upstream environment setup failed for {args.task}, seed {seed}")
    return success, steps, "", ""


def navigation_episode(args, seed):
    from jarvisvla.evaluate.agent_wrapper import VLLM_AGENT
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import CommandsCallback
    from minestudio.simulator.entry import CameraConfig

    class GoldPillar(CommandsCallback):
        def after_reset(self, sim, obs, info):
            pose = info["player_pos"]
            x, y, z = (math.floor(pose[key]) for key in ("x", "y", "z"))
            self.start = (x + 0.5, z + 0.5)
            self.target = (x + 0.5, z + 5.5)
            self.commands = [
                f"/fill {x-4} {y-1} {z-4} {x+4} {y-1} {z+9} minecraft:grass_block",
                f"/fill {x-4} {y} {z-4} {x+4} {y+4} {z+9} minecraft:air",
                f"/fill {x} {y} {z+5} {x} {y+2} {z+5} minecraft:gold_block",
                f"/tp @p {self.start[0]} {y} {self.start[1]} 0 0",
                f"/execute if block {x} {y+1} {z+5} minecraft:gold_block run give @p minecraft:paper 1",
            ]
            return super().after_reset(sim, obs, info)

    scene = GoldPillar([])
    sim = MinecraftSim(
        action_type="env", seed=seed, preferred_spawn_biome="plains", callbacks=[scene],
        obs_size=(640, 360), render_size=(640, 360),
        camera_config=CameraConfig(camera_binsize=1, camera_maxval=10, camera_mu=20,
                                   camera_quantization_scheme="mu_law"),
    )
    try:
        _, info = sim.reset()
        for _ in range(6):
            _, _, terminated, truncated, info = sim.step(sim.noop_action())
            if terminated or truncated:
                raise RuntimeError("The simulator ended during navigation setup")
        pose = info["player_pos"]
        if math.dist((pose["x"], pose["z"]), scene.start) > 0.2:
            raise RuntimeError(f"Unexpected navigation starting pose: {pose}")
        if abs((pose["yaw"] + 180) % 360 - 180) > 1 or abs(pose["pitch"]) > 1:
            raise RuntimeError(f"Unexpected navigation starting heading: {pose}")
        if not any(item["type"].split(":")[-1] == "paper"
                   for item in info["inventory"].values()):
            raise RuntimeError("Navigation gold pillar check yielded no marker")
        sim.env.execute_cmd("/clear @p minecraft:paper")
        for _ in range(2):
            _, _, terminated, truncated, info = sim.step(sim.noop_action())
            if terminated or truncated:
                raise RuntimeError("The simulator ended before navigation control")
        sim.action_type = "agent"
        agent = VLLM_AGENT(checkpoint_path=str(args.checkpoint), base_url=args.base_url,
                           history_num=2, action_chunk_len=1, instruction_type="normal",
                           temperature=0.9)
        closest = math.dist(scene.start, scene.target)
        trace_path = args.output / f"navigation-{seed}.csv"
        with trace_path.open("x", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("step", "x", "z", "yaw", "pitch", "dist", "buttons", "camera"))
            for step in range(1, NAVIGATION_LIMIT + 1):
                action = agent.forward([info["pov"]], [NAVIGATION_INSTRUCTION])
                buttons, camera = int(action["buttons"]), int(action["camera"])
                _, _, terminated, truncated, info = sim.step(action)
                pose = info["player_pos"]
                distance = math.dist((pose["x"], pose["z"]), scene.target)
                closest = min(closest, distance)
                writer.writerow((step, pose["x"], pose["z"], pose["yaw"], pose["pitch"],
                                 distance, buttons, camera))
                output.flush()
                if distance <= NAVIGATION_RADIUS or terminated or truncated:
                    break
        return distance <= NAVIGATION_RADIUS, step, distance, closest
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=(*PRESETS, NAVIGATION))
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--start-seed", type=int, default=19961103)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    episode = navigation_episode if args.task == NAVIGATION else official_episode
    with (args.output / "episodes.csv").open("x", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("task", "seed", "success", "steps", "final_dist", "min_dist",
                         "elapsed_seconds"))
        output.flush()
        for seed in range(args.start_seed, args.start_seed + args.count):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            initialized = perf_counter()
            result = episode(args, seed)
            writer.writerow((args.task, seed, *result, perf_counter() - initialized))
            output.flush()
            print(args.task, seed, result, flush=True)


if __name__ == "__main__":
    main()
