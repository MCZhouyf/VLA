"""Apply the randomized-heading acceptance gate to retained trial records."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from phase1_metrics import wilson
from phase1_t1 import DISTANCES, TEACHER_SEED


def summarize(directories):
    records = {}
    for directory in directories:
        for path in sorted(directory.glob("worker-*/acceptance.csv")):
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    key = int(row["distance"]), int(row["seed"])
                    if key in records and not records[key]["error"]:
                        raise ValueError(f"Duplicate completed T1 acceptance case {key}")
                    records[key] = row
    expected = {(distance, seed) for distance in DISTANCES
                for seed in range(TEACHER_SEED, TEACHER_SEED + 50)}
    if not records.keys() <= expected:
        raise ValueError("Unexpected T1 acceptance seed or distance")
    summaries = []
    for distance in DISTANCES:
        subset = [row for (d, _), row in records.items() if d == distance]
        usable = [row for row in subset if not row["error"]]
        old = sum(row["v3_shortcut_success"] == "True" for row in usable)
        new = sum(row["v4_shortcut_success"] == "True" for row in usable)
        successes = len(usable)
        guarded = sum(bool(row.get("v3_shortcut_end") and row.get("v4_shortcut_end")) for row in usable)
        low, high = wilson(successes, 50)
        summaries.append(dict(
            distance=distance, completed=len(subset), teacher_successes=successes,
            teacher_success_rate=successes / 50, teacher_ci95_low=low, teacher_ci95_high=high,
            teacher_mean_steps=statistics.fmean(int(r["teacher_steps"]) for r in usable) if usable else None,
            shortcut_tests=successes, guarded_shortcut_tests=guarded,
            v3_shortcut_successes=old, v4_shortcut_successes=new,
            accepted=len(subset) == 50 and successes >= 48 and guarded == successes and
            (distance == 10 or new / successes < 0.10),
        ))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    rows = summarize(args.directories)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = all(row["accepted"] for row in rows)
    print(json.dumps(rows, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
