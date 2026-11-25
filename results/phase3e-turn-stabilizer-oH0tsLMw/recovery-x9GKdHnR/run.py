"""Complete missing stabilized T1 v3 oracle evaluations on GPUs 6 and 7."""

import argparse
import csv
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep


DISTANCES = (10, 30, 60, 100)
INITIAL = ("recheck-10-gpu6", "recheck-10-gpu7", "recheck-long-gpu6", "recheck-long-gpu7")


def read_scores(directory, expected):
    path = directory / "episodes.csv"
    result = {}
    if not path.is_file():
        return result
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = int(row["d"]), int(row["seed"])
            if key not in expected or key in result or row["success"] not in ("True", "False"):
                raise ValueError(f"Invalid evaluation row {key} in {path}")
            result[key] = dict(row, source=str(directory))
    return result


def extend_scores(merged, additional):
    if merged.keys() & additional.keys():
        raise ValueError("Duplicate frozen evaluations")
    merged.update(additional)


def recover(gpu, pending, entry, original, output, env):
    results = {}
    for attempt in range(1, 4):
        if not pending:
            return results
        plans = defaultdict(list)
        for seed in sorted({seed for _, seed in pending}):
            plans[tuple(d for d in DISTANCES if (d, seed) in pending)].append(seed)
        for part, (distances, seeds) in enumerate(plans.items(), 1):
            tag = f"gpu{gpu}-attempt{attempt}-part{part}"
            log = output / f"{tag}.log"
            directory = output / tag
            command = [
                "bash", str(entry), "waypoint-eval", str(gpu), str(directory),
                "--training-run", str(original / "controller"), "--metrics", str(original / "frozen/metrics.json"),
                "--fixtures", str(original / "frozen/fixtures.json"), "--batch", "3", "--stabilize-turns",
                "--distances", *map(str, distances), "--seeds", *map(str, seeds),
            ]
            with log.open("x") as stream:
                status = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            expected = {(distance, seed) for distance in distances for seed in seeds}
            scored = read_scores(directory, expected)
            extend_scores(results, scored)
            pending.difference_update(scored)
            print(f"{tag} status={status} scored={len(scored)}/{len(expected)}", flush=True)
            if scored.keys() == expected:
                continue
            errors = log.read_text(errors="replace")
            if status != 75 and "T1 /execute if block validation failed" not in errors and "GLFW" not in errors:
                raise RuntimeError(f"Unresolved run failure in {log}")
            sleep(30)
    if pending:
        raise RuntimeError(f"GPU {gpu} still has {len(pending)} unscored evaluations")
    return results


def run(recheck, output):
    original = Path((recheck / "original-run").read_text().strip())
    fixtures = json.loads((original / "frozen/fixtures.json").read_text())
    seeds = fixtures["formal"]
    if len(seeds) != 30 or len(set(seeds)) != 30:
        raise ValueError("Expected 30 distinct frozen seeds")
    expected = {(distance, seed) for distance in DISTANCES for seed in seeds}
    results = {}
    for part in INITIAL:
        extend_scores(results, read_scores(recheck / part, expected))
    pending = expected - results.keys()
    if not pending or any(distance not in (30, 60) for distance, _ in pending):
        raise ValueError("Unexpected incomplete frozen coverage")
    missing = sorted({seed for _, seed in pending})
    plan = {6: missing[9:], 7: missing[:9]}
    (output / "recovery_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    entry = recheck / "snapshot/code/minemem/t1_exec.sh"
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = []
        for gpu, assigned in plan.items():
            if assigned:
                env = dict(os.environ, TZ="Asia/Shanghai", WORK=str(recheck.parents[1]),
                           T1_LOCAL_DATABASE_ROOT=tempfile.mkdtemp(dir="/dev/shm", prefix="minemem-t1-recovery-"))
                jobs.append(pool.submit(recover, gpu, {key for key in pending if key[1] in assigned},
                                        entry, original, output, env))
        for job in jobs:
            extend_scores(results, job.result())
    if results.keys() != expected:
        raise ValueError("Incomplete recovered frozen coverage")
    scores = {distance: sum(results[distance, seed]["success"] == "True" for seed in seeds)
              for distance in DISTANCES}
    with (output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(results.values()))))
        writer.writeheader()
        writer.writerows(results[key] for key in sorted(results))
    passed = all(count >= 24 for count in scores.values())
    gate = dict(passed=passed, distances={distance: dict(successes=count, n=30, rate=count/30)
                                          for distance, count in scores.items()})
    (output / "recheck_oracle_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recheck", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    status = 1
    try:
        status = run(args.recheck.resolve(), args.output.resolve())
    finally:
        (args.output / "recovery.status").write_text(f"{status}\n")
    raise SystemExit(status)
