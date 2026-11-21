"""Record a MineStudio simulator smoke test or a one-GPU MCU rollout."""

import argparse
import csv
import json
import os
import random
from collections import deque
from math import floor
from pathlib import Path
from time import perf_counter


class JarvisPolicy:
    def __init__(self, settings, seed, model_dir):
        import torch
        from jarvisvla.inference.action_mapping import OneActionTokenizer
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if torch.cuda.device_count() != 1:
            raise RuntimeError("MCU rollout requires exactly one visible GPU")
        torch.manual_seed(seed)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.actions = OneActionTokenizer(tokenizer_type="qwen2_vl")
        self.settings = settings
        self.history = deque(maxlen=settings["history"])

    def __call__(self, frame):
        import torch
        from PIL import Image

        current = Image.fromarray(frame)
        history = list(self.history)
        if not history:
            history = [(current, self.actions.null_token())] * self.settings["history"]
        entries = [*history, (current, None)]
        messages = []
        for index, (image, action_text) in enumerate(entries):
            prefix = self.settings["instruction"] if index == 0 else ""
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"{prefix}\nobservation: "},
                {"type": "image", "image": image},
            ]})
            if action_text is not None:
                messages.append({"role": "assistant", "content": action_text})
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=[prompt], images=[image for image, _ in entries],
                                return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.settings["max_new_tokens"], do_sample=False,
            )
        ids = generated[0, inputs.input_ids.shape[-1]:].tolist()
        try:
            start = ids.index(self.actions.act_beg_id)
            end = ids.index(self.actions.act_end_id, start + 1)
        except ValueError:
            return None, ids
        action_ids = ids[start:end + 1]
        action_text = self.processor.tokenizer.decode(action_ids, skip_special_tokens=False)
        self.history.append((current, action_text))
        return self.actions.decode(action_ids)[0], ids


def make_sim(mode, config):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import CommandsCallback, RewardsCallback
    from minestudio.simulator.entry import CameraConfig

    if mode == "random":
        class ResetCommands(CommandsCallback):
            def after_reset(self, sim, obs, info):
                pose = info["player_pos"]
                x, y, z = (floor(pose[key]) for key in ("x", "y", "z"))
                coordinates = dict(x=x, x1=x + 1, x2=x + 2, y=y + 4, z=z)
                self.commands = [template.format(**coordinates)
                                 for template in config["random_commands"]]
                return super().after_reset(sim, obs, info)

        return MinecraftSim(
            action_type="env", seed=config["seed"], preferred_spawn_biome="plains",
            callbacks=[ResetCommands([])],
        )

    settings = config["mcu"]
    resolution = tuple(settings["resolution"])
    inventory = {
        0: {"type": "stone_shovel", "quantity": 1},
        1: {"type": "stone_pickaxe", "quantity": 1},
    }
    callbacks = []
    if settings.get("scene") == "visible_stone":
        class StoneScene(CommandsCallback):
            def after_reset(self, sim, obs, info):
                pose = info["player_pos"]
                x, y, z = (floor(pose[key]) for key in ("x", "y", "z"))
                self.commands = [
                    f"/fill {x-1} {y-1} {z} {x+1} {y-1} {z+2} minecraft:cobblestone",
                    f"/fill {x-1} {y} {z} {x+1} {y+2} {z+1} minecraft:air",
                    f"/fill {x-1} {y} {z+2} {x+1} {y+2} {z+2} minecraft:stone",
                    f"/tp @p {x+0.5} {y} {z+0.5} 0 0",
                ]
                return super().after_reset(sim, obs, info)

        inventory[0], inventory[1] = inventory[1], inventory[0]
        callbacks.append(StoneScene([]))
    callbacks.append(RewardsCallback([{
        "event": "mine_block", "objects": ["stone"], "identity": "mine stone",
        "reward": 1.0, "max_reward_times": 1,
    }]))
    return MinecraftSim(
        action_type="agent", seed=config["seed"], preferred_spawn_biome="forest",
        obs_size=resolution, render_size=resolution,
        camera_config=CameraConfig(**settings["camera"]),
        inventory=inventory, callbacks=callbacks,
    )


def check_random_reset(sim, info):
    import numpy as np
    from minestudio.utils.vpt_lib.actions import Buttons

    pose = info["player_pos"]
    item_types = {slot["type"].split(":")[-1] for slot in info["inventory"].values()}
    voxels = info.get("voxels", [])
    block_types = [voxel["type"].split(":")[-1] for voxel in voxels]
    sample = {key: np.array([key == "forward"]) for key in Buttons.ALL}
    sample["camera"] = np.array([[2.0, -3.0]], dtype=np.float32)
    agent = sim.env_action_to_agent_action(sample)
    reconstructed = sim.agent_action_to_env_action(agent)
    round_trip = sim.env_action_to_agent_action(reconstructed)
    yaw = (float(pose["yaw"]) + 180) % 360 - 180
    checks = {
        "tp": abs(yaw - 45) < 1 and abs(float(pose["pitch"])) < 1,
        "give": "emerald" in item_types,
        "setblock": "gold_block" in block_types or "diamond" in item_types,
        "fill": block_types.count("iron_block") == 2 or "lapis_block" in item_types,
        "action_round_trip": all(np.array_equal(agent[key], round_trip[key])
                                 for key in ("buttons", "camera")),
    }
    return checks, {"player_pos": pose, "inventory_types": sorted(item_types), "voxels": voxels}


def run(mode, config_path, output, scene="natural"):
    import cv2
    import numpy as np
    import yaml

    config = yaml.safe_load(config_path.read_text())
    visible_stone = mode == "mcu" and scene == "visible_stone"
    config["mcu"]["scene"] = scene
    if visible_stone:
        config["mcu"]["max_steps"] = config["mcu"]["visible_stone"]["max_steps"]
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    mcu = mode == "mcu"
    policy = None
    if mcu:
        model_dir = Path(os.environ["WORK"]) / "models" / "JarvisVLA-Qwen2-VL-7B"
        policy = JarvisPolicy(config["mcu"], seed, model_dir)
    limit = config["mcu"]["max_steps"] if mcu else config["random_steps"]
    sim = make_sim(mode, config)
    success = False
    stop = "step_limit"
    checks = {}
    steps = 0
    loop_seconds = 0.0
    writer = None
    try:
        obs, info = sim.reset()
        if visible_stone:
            for _ in range(config["mcu"]["visible_stone"]["setup_steps"]):
                action = sim.noop_action()
                action["camera"] = np.array([sim.action_mapper.camera_null_idx])
                obs, _, terminated, truncated, info = sim.step(action)
                if terminated or truncated:
                    raise RuntimeError("The simulator ended during the stone scene setup")
            (output / "reset_evidence.json").write_text(json.dumps({
                "commands": sim.callbacks[0].commands, "player_pos": info["player_pos"],
            }, default=float))
        if not mcu:
            samples = []
            for _ in range(config["random_probe_steps"]):
                probe = sim.noop_action()
                probe["voxels"] = config["random_voxel_bounds"]
                obs, _, terminated, truncated, info = sim.step(probe)
                if terminated or truncated:
                    raise RuntimeError("The simulator ended during reset verification")
                checks, evidence = check_random_reset(sim, info)
                samples.append(evidence)
                if all(checks.values()):
                    break
            (output / "reset_evidence.json").write_text(json.dumps({
                "commands": sim.callbacks[0].commands, "samples": samples,
            }, default=float))
        height, width = obs["image"].shape[:2]
        writer = cv2.VideoWriter(str(output / "rollout.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 config["video_fps"], (width, height))
        if not writer.isOpened():
            raise RuntimeError("Cannot open MP4 writer")
        writer.write(cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR))
        action_space = sim.action_space
        action_space.seed(seed)
        with (output / "steps.csv").open("w", newline="") as stream:
            rows = csv.writer(stream)
            rows.writerow(["step", "loop_seconds", "reward", "x", "y", "z",
                           "yaw", "pitch", "generated_token_ids"])
            for _ in range(limit):
                start = perf_counter()
                if mcu:
                    action, tokens = policy(info["pov"])
                else:
                    action, tokens = action_space.sample(), []
                if action is None:
                    rows.writerow([steps + 1, perf_counter() - start, "", "", "",
                                   "", "", "", json.dumps(tokens)])
                    stop = "missing_action_tags"
                    break
                obs, reward, terminated, truncated, info = sim.step(action)
                elapsed = perf_counter() - start
                loop_seconds += elapsed
                steps += 1
                pose = info["player_pos"]
                rows.writerow([steps, elapsed, reward,
                               *(pose[key] for key in ("x", "y", "z", "yaw", "pitch")),
                               json.dumps(tokens)])
                writer.write(cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR))
                if mcu and reward > 0:
                    success, stop = True, "mined_stone"
                    break
                if terminated or truncated:
                    stop = "environment_ended"
                    break
    finally:
        if writer is not None:
            writer.release()
        sim.close()
    if not mcu:
        success = steps == limit and all(checks.values())
    summary = {
        "mode": mode, "seed": seed, "success": success, "stop": stop, "steps": steps,
        "fps": steps / loop_seconds if steps else "", **checks,
    }
    if mcu:
        summary["scene"] = scene
        summary["gpu_index"] = os.environ.get("MINEMEM_ACTIVE_GPU", "")
    with (output / "summary.csv").open("w", newline="") as stream:
        rows = csv.DictWriter(stream, fieldnames=list(summary))
        rows.writeheader()
        rows.writerow(summary)
    print(summary)
    return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("random", "mcu"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene", choices=("natural", "visible_stone"), default="natural")
    args = parser.parse_args()
    if args.mode != "mcu" and args.scene != "natural":
        parser.error("--scene is available only for MCU")
    raise SystemExit(run(args.mode, Path(__file__).with_suffix(".yaml"), args.output, args.scene))
