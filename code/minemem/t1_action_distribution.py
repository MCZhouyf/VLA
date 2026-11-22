"""Count the decoded return actions selected by the frozen training index."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

from jarvisvla.inference.action_mapping import OneActionTokenizer
from minestudio.utils.vpt_lib.actions import Buttons
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    actions = OneActionTokenizer(tokenizer_type="qwen2_vl")

    @lru_cache
    def decode(action):
        ids = tokenizer.encode(action, add_special_tokens=False)
        if ids[0] != actions.act_beg_id or ids[-1] != actions.act_end_id:
            raise ValueError("Malformed teacher action")
        decoded, = actions.decode(ids)
        factored = actions.action_mapper.to_factored(decoded)
        return factored["buttons"].tolist(), factored["camera"].tolist()

    totals = Counter()
    counts = defaultdict(Counter)
    for line in args.dataset.read_text().splitlines():
        row = json.loads(line)
        if row["split"] != "train" or not row["selected_labels"]:
            continue
        with ZipFile(row["path"]) as episode:
            selected = json.loads(episode.read("episode.json"))["action"][row["outbound_frames"]:row["selected_frames"]]
        if len(selected) != row["selected_labels"]:
            raise ValueError(f"Return label count differs for {row['scene_seed']}")
        for name in ("all", str(row["group"])):
            totals[name] += len(selected)
            for action, occurrences in Counter(selected).items():
                buttons, (pitch, yaw) = decode(action)
                for button, active in zip(Buttons.ALL, buttons, strict=True):
                    counts[name]["button", button] += occurrences * int(active)
                counts[name]["pitch_bin", pitch] += occurrences
                counts[name]["yaw_bin", yaw] += occurrences

    with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("group","kind","value","count","actions","fraction"))
        for group, total in totals.items():
            for kind, values in (("button", Buttons.ALL), ("pitch_bin", range(actions.n_camera_bins)),
                                 ("yaw_bin", range(actions.n_camera_bins))):
                for value in values:
                    count = counts[group][kind, value]
                    writer.writerow((group, kind, value, count, total, count/total))


if __name__ == "__main__":
    main()
