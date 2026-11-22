"""Evaluate a saved T1 checkpoint against the entire frozen validation split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from t1_train import VALIDATION_FIELDS, validation_metrics
from t1_trained_eval import Controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers:
        parser.error("Invalid validation shard")
    summary = json.loads((args.training_run/"summary.json").read_text())
    settings = json.loads((args.training_run/"training_config.json").read_text())
    source = args.dataset.read_bytes()
    if hashlib.sha256(source).hexdigest() != settings["dataset_sha256"]:
        raise ValueError("Validation requires the same frozen index as training")
    rows = [json.loads(line) for line in source.splitlines()]
    validation = [row for row in rows if row["split"]=="validation"]
    if sum(row["selected_frames"] for row in validation) != 6500:
        raise ValueError("Validation split must contain 6,500 frames")
    validation = validation[args.worker::args.workers]
    policy = Controller(args.checkpoint, args.training_run, args.device).policy
    result = validation_metrics(policy, validation, settings["seed"], settings["resolution"],
                                summary["updates"])
    with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(VALIDATION_FIELDS)
        writer.writerow(result)
    print(dict(zip(VALIDATION_FIELDS, result, strict=True)))


if __name__ == "__main__":
    main()
