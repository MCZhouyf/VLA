"""Decode the first action executed by Phase 0 and count buttons and camera bins."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from jarvisvla.inference.action_mapping import OneActionTokenizer
from minestudio.utils.vpt_lib.actions import Buttons


def audit(path, tokenizer):
    counts = Counter()
    steps = 0
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if not row["reward"]:
                continue
            tokens = json.loads(row["generated_token_ids"])
            start = tokens.index(tokenizer.act_beg_id)
            end = tokens.index(tokenizer.act_end_id, start + 1)
            action, = tokenizer.decode(tokens[start:end + 1])
            factored = tokenizer.action_mapper.to_factored(action)
            for button, pressed in zip(Buttons.ALL, factored["buttons"], strict=True):
                counts["button", button] += int(pressed)
            pitch, yaw = factored["camera"].tolist()
            counts["pitch_bin", pitch] += 1
            counts["yaw_bin", yaw] += 1
            counts["camera_pair", f"{pitch},{yaw}"] += 1
            steps += 1
    if not steps:
        raise ValueError(f"No executed actions in {path}")
    options = {
        "button": Buttons.ALL,
        "pitch_bin": range(tokenizer.n_camera_bins),
        "yaw_bin": range(tokenizer.n_camera_bins),
        "camera_pair": sorted(value for kind, value in counts if kind == "camera_pair"),
    }
    for kind, values in options.items():
        for value in values:
            count = counts[kind, value]
            yield path.parent.name, kind, value, count, steps, count / steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("steps", nargs="+", type=Path)
    args = parser.parse_args()
    tokenizer = OneActionTokenizer(tokenizer_type="qwen2_vl")
    with args.output.open("x", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("run", "kind", "value", "count", "steps", "fraction"))
        for path in args.steps:
            writer.writerows(audit(path, tokenizer))


if __name__ == "__main__":
    main()
