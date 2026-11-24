"""Audit the action round trip and previous-action baseline on frozen T1 labels."""

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from jarvisvla.inference.action_mapping import OneActionTokenizer
from minestudio.simulator.entry import CameraConfig, MinecraftSim
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.utils.vpt_lib.actions import ActionTransformer
from transformers import AutoTokenizer

from phase1_t1 import CAMERA


def audit(dataset, checkpoint, archive_cache):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    actions = OneActionTokenizer(tokenizer_type="qwen2_vl")
    config = CameraConfig(**CAMERA)
    sim = MinecraftSim.__new__(MinecraftSim)
    sim.action_mapper = CameraHierarchicalMapping(n_camera_bins=config.n_camera_bins)
    sim.action_transformer = ActionTransformer(**config.action_transformer_kwargs)

    def decoded(text):
        return actions.decode(tokenizer.encode(text, add_special_tokens=False))[0]

    @lru_cache
    def round_trip(label):
        action = decoded(label)
        env = sim.agent_action_to_env_action(action)
        recovered = sim.env_action_to_agent_action({key: np.asarray([val]) for key, val in env.items()})
        text = actions.encode_action((int(recovered["buttons"].item()), int(recovered["camera"].item())))
        returned = sim.agent_action_to_env_action(decoded(text))
        return (
            text != label,
            not all(np.array_equal(env[key], returned[key]) for key in env),
        )

    middle = config.n_camera_bins // 2
    camera_bins = []
    for yaw in range(config.n_camera_bins):
        text = actions.encode_action((0, middle * config.n_camera_bins + yaw))
        env = sim.agent_action_to_env_action(decoded(text))
        camera_bins.append(np.asarray(env["camera"]).flatten().tolist())
    assert camera_bins[middle] == [0.0, 0.0]
    assert camera_bins[0] == [0.0, -10.0] and camera_bins[-1] == [0.0, 10.0]

    train = Counter()
    validation = Counter()
    validation_groups = {}
    for line in dataset.read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "test" or not row["selected_labels"]:
            continue
        path = Path(row["path"])
        if archive_cache:
            path = archive_cache / path.name
        with ZipFile(path) as episode:
            meta = json.loads(episode.read("episode.json"))
        labels = meta["action"]
        start, stop = row["outbound_frames"], row["selected_frames"]
        if len(labels[start:stop]) != row["selected_labels"]:
            raise ValueError(f"Incomplete selected labels for {row['scene_seed']}")
        if row["split"] == "train":
            for label, count in Counter(labels[start:stop]).items():
                token_error, env_error = round_trip(label)
                train["actions"] += count
                train["token_mismatches"] += count * token_error
                train["environment_mismatches"] += count * env_error
        else:
            group = validation_groups.setdefault(str(row["group"]), Counter())
            for index in range(start, stop):
                previous = labels[index - 1] if index else actions.null_token()
                same = labels[index] == previous
                for stats in (validation, group):
                    stats["actions"] += 1
                    stats["copy_exact"] += same
                    stats["changed"] += not same
                    stats["initial_return_actions"] += index == start
                    stats["initial_copy_exact"] += same and index == start
    return dict(train_roundtrip=dict(train), unique_train_labels=round_trip.cache_info().currsize,
                center_yaw_bin=middle, yaw_bins_as_pitch_yaw=camera_bins,
                validation_previous_action=dict(validation), validation_groups=validation_groups,
                source="Frozen serialized labels, decoded to the environment action grid before round trip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset, args.checkpoint, args.archive_cache)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
