"""Compare the training and rollout boundary on identical navigation scenes."""

import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from PIL import Image

from t1_collect import write_episode
from t1_train import case_examples, load_images
from t1_v4 import VISIBLE_INSTRUCTION


def action_record(arena, agent):
    action = arena.sim.agent_action_to_env_action(agent)
    pitch, yaw = action["camera"]
    return dict(buttons=int(agent["buttons"]), camera=int(agent["camera"]),
                pitch=float(pitch), yaw=float(yaw), forward=int(action["forward"]))


def prompt_record(policy, example, images, raw_sizes):
    batch = policy.prompts(*example[:3])
    ids = batch["input_ids"][0].tolist()
    vision = policy.processor.image_processor(images=images, return_tensors="pt")
    return dict(
        instruction=example[0], history_actions=example[2], input_ids=ids,
        template=policy.processor.tokenizer.decode(ids, skip_special_tokens=False),
        image_grid_thw=batch["image_grid_thw"].tolist(), raw_rgb_sizes=raw_sizes,
        wrapped_rgb_sizes=[list(image.size) for image in images],
        pixel_values_shape=list(vision["pixel_values"].shape),
        pixel_values_sha256=hashlib.sha256(vision["pixel_values"].numpy().tobytes()).hexdigest(),
    )


@torch.inference_mode()
def inspect_start(output, arena, controller, config, outbound, teacher_return, cohort):
    policy = controller.policy
    resolution = controller.settings.get("resolution")
    seed = config["scenario_seed"]
    with TemporaryDirectory(dir=output, prefix=".probe-") as tmp:
        folder = Path(tmp)
        row = write_episode(folder, config, outbound, teacher_return[:1], "train", "visible")
        archive = folder / row["path"]
        row.update(path=str(archive), selected_frames=row["frames"], selected_labels=1)
        example, = case_examples(policy, row, 0, resolution, controller.cache_vision)
        with ZipFile(archive) as source:
            images = load_images(source, range(row["frames"]), policy.images, resolution)
            raw_sizes = []
            for index in range(row["frames"]):
                with Image.open(io.BytesIO(source.read(f"{index:05d}.jpg"))) as image:
                    raw_sizes.append(list(image.size))
        instruction, frames, actions, expected, features, poses = example
        train_input = (instruction, frames, actions, features, poses)
        train_prompt = prompt_record(policy, train_input, images, raw_sizes)

    arena.restore_return_pose(config)
    controller.start(outbound)
    live_pixels = arena.info["pov"]
    current_rgb = Image.fromarray(live_pixels)
    current = controller.image(current_rgb)
    current_frame = policy.encode_frames([current])[0] if controller.cache_vision else current
    live, _ = controller.request(arena.info, VISIBLE_INSTRUCTION, current_frame)
    live_prompt = prompt_record(policy, live, [*images[:-1], current],
                                [*raw_sizes[:-1], list(current_rgb.size)])
    raw_processor = policy.processor.image_processor.to_dict()
    wrapper = {key: getattr(policy.images, key) for key in
               ("model_name", "image_factor", "min_pixels", "max_pixels", "max_ratio")}
    keys = [key for key in train_prompt if train_prompt[key] != live_prompt[key]]
    positions = [i for i, (a, b) in enumerate(zip(train_prompt["input_ids"], live_prompt["input_ids"]))
                 if a != b]
    first_pose = teacher_return[0][1]
    live_pose = [float(arena.info["player_pos"][key]) for key in ("x", "z", "yaw")]
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(live_pixels, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise RuntimeError("Could not reproduce the teacher JPEG preprocessing")
    with Image.open(io.BytesIO(jpeg.tobytes())) as decoded:
        jpeg_image = controller.image(decoded.convert("RGB"))
    jpeg_frame = policy.encode_frames([jpeg_image])[0] if controller.cache_vision else jpeg_image
    jpeg_live, _ = controller.request(arena.info, VISIBLE_INSTRUCTION, jpeg_frame)
    forward = arena.sim.noop_action()
    forward["forward"] = 1
    forward = arena.sim.env_action_to_agent_action({key: np.asarray([val]) for key,val in forward.items()})
    forward_text = policy.actions.encode_action((int(forward["buttons"].item()), int(forward["camera"].item())))
    probes = {
        "original_history": live,
        "empty_history": (*live[:2], [""] * policy.history, *live[3:]),
        "forward_history": (*live[:2], [forward_text] * policy.history, *live[3:]),
        "teacher_jpeg_same_pose": train_input,
        "live_rgb_via_jpeg86": jpeg_live,
    }
    expected_ids = policy.action_ids(expected)
    results = {}
    for name, inputs in probes.items():
        agent, text, tags, ids = policy.predict(*inputs, return_ids=True)
        results[name] = dict(raw_token_ids=ids, raw_text=text, complete=bool(tags),
                             exact=ids == expected_ids, **action_record(arena, agent))
    summary = dict(seed=seed, cohort=cohort, expected_ids=expected_ids,
                   expected=action_record(arena, policy.actions.decode(expected_ids)[0]),
                   original_history=live[2], controls=results, differing_input_fields=keys,
                   teacher_pose=first_pose, live_pose=live_pose)
    with (output / "start-probes.jsonl").open("a") as stream:
        stream.write(json.dumps(summary) + "\n")
    with (output / f"seed{seed}-input-diff.json").open("x") as stream:
        json.dump(dict(
            seed=seed, training=train_prompt, rollout=live_prompt, differing_fields=keys,
            differing_token_positions=positions, processor=raw_processor, image_wrapper=wrapper,
            resize_override=resolution, cache_vision=controller.cache_vision,
            teacher_pose=first_pose, live_pose=live_pose,
        ), stream, indent=2)
        stream.write("\n")
