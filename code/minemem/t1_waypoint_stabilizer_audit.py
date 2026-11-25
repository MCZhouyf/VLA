"""Locate the first intervention the waypoint stabilizer would make on original traces."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from t1_waypoint import WaypointTurnStabilizer


def first_intervention(path, total_steps):
    guard = WaypointTurnStabilizer()
    with path.open() as stream:
        actions = [json.loads(line) for line in stream]
    if len(actions) != total_steps or any("control_override" in action for action in actions):
        raise ValueError(f"Expected a complete original action trace in {path}")
    for action in actions:
        proposed = {**action["buttons"], "camera": (action["pitch"], action["yaw"])}
        if guard.should_advance(action["instruction"], proposed):
            return dict(step=action["step"], waypoint=action["instruction"].splitlines()[-1],
                        policy_yaw=action["yaw"])
    return None


def audit(run):
    results = []
    counts = Counter()
    for gpu in (6, 7):
        directory = run / f"oracle-gpu{gpu}"
        with (directory / "episodes.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            distance, seed = int(row["d"]), int(row["seed"])
            success = row["success"] == "True"
            intervention = first_intervention(directory / f"d{distance}-seed{seed}-actions.jsonl",
                                              int(row["steps"]))
            outcome = "successful" if success else "failed"
            counts[outcome] += 1
            if intervention is not None:
                counts[f"{outcome}_with_override"] += 1
            results.append(dict(distance=distance, seed=seed, original_success=success,
                                original_reason=row["end_reason"], first_intervention=intervention))
    return dict(counts=dict(counts), episodes=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.run), indent=2) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        with args.output.open("x") as output:
            output.write(result)
