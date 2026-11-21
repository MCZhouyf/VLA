"""Aggregate generation checks and T1 teacher results without discarding failures."""

import argparse
import csv
import math
import statistics
from pathlib import Path


PUBLISHED = {"craft/craft_crafting_table": 0.87, "smelt/iron_ingot": 0.70}
START_CHECK_SEED = 19961103
START_TEACHER_SEED = 2026112100


def read_rows(paths):
    for path in paths:
        with path.open(newline="") as source:
            yield from csv.DictReader(source)


def wilson(successes, total):
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - spread)
    upper = 1.0 if successes == total else min(1.0, center + spread)
    return lower, upper


def summarize(rows, kind, expected, start_seed, accept):
    seed_ids = [int(row["seed"]) for row in rows]
    allowed = set(range(start_seed, start_seed + expected))
    if len(seed_ids) != len(set(seed_ids)) or not set(seed_ids).issubset(allowed):
        raise ValueError(f"Invalid or duplicate seed record for {kind}: {seed_ids}")
    if any(row["success"] not in ("True", "False") for row in rows):
        raise ValueError(f"Invalid success marker for {kind}")
    successes = [row for row in rows if row["success"] == "True"]
    total = len(rows)
    rate = len(successes) / total if total else None
    lo, hi = wilson(len(successes), total) if total else (None, None)
    mean_steps = statistics.fmean(int(row["steps"]) for row in successes) if successes else None
    return dict(group=kind, completed=total, expected=expected, successes=len(successes),
                success_rate=rate, ci95_low=lo, ci95_high=hi, mean_success_steps=mean_steps,
                accepted=total == expected and accept(rate))


def gate(paths):
    raw = list(read_rows(paths))
    for task in (*PUBLISHED, "navigation"):
        reference = PUBLISHED.get(task)
        rows = [row for row in raw if row["task"] == task]
        result = summarize(
            rows, task, 10, START_CHECK_SEED,
            lambda rate: abs(rate - reference) <= 0.150000001 if reference is not None
            else rate >= 0.6,
        )
        rate = result["success_rate"]
        difference = None if rate is None or reference is None else rate - reference
        yield {**result, "paper_success_rate": reference, "difference": difference}


def teacher(paths, seeds):
    raw = list(read_rows(paths))
    for distance in (10, 30, 60, 100):
        rows = [row for row in raw if int(row["distance"]) == distance]
        yield summarize(rows, distance, seeds, START_TEACHER_SEED, lambda rate: rate >= 0.95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("gate", "teacher"))
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--teacher-seeds", type=int, default=50)
    args = parser.parse_args()
    rows = list(gate(args.inputs) if args.kind == "gate" else teacher(args.inputs, args.teacher_seeds))
    with args.output.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["accepted"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
