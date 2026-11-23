"""Freeze the validated second T1 geometry for trained-policy evaluation."""

import argparse
import csv
import json
from pathlib import Path

from t1_occlusion import SCREENED_PROTOCOL
from t1_safety import SETUP_REVISION
from t1_v4 import DISTANCES


def freeze(scores, seed_file):
    frozen = json.loads(seed_file.read_text())
    worlds = frozen["world_seed_by_scenario"]
    with scores.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    grid = {(int(row["d"]), row["seed"]) for row in records}
    if grid != {(d, seed) for d in DISTANCES for seed in worlds} or len(grid) != len(records):
        raise ValueError("Screened acceptance must contain one row per frozen seed and distance")
    fixtures = {}
    formal = set(frozen["formal"])
    for distance in DISTANCES:
        group = [row for row in records if int(row["d"]) == distance]
        teachers = sum(row["teacher_success"] == "True" for row in group)
        if teachers < 48:
            raise ValueError(f"Teacher missed the screened acceptance gate at {distance}")
        for policy in ("turn180_success", "success"):
            if distance >= 30 and (sum(row[policy] == "True" for row in group) >= 5 or
                                   sum(row[policy] == "True" and int(row["seed"]) in formal
                                       for row in group) >= 3):
                raise ValueError(f"Screened shortcut gate failed for {policy} at {distance}")
        for row in group:
            seed = row["seed"]
            if int(seed) not in formal:
                continue
            source = scores.parent / row["source"]
            config = json.loads((source.parent / f"d{distance}-seed{seed}.json").read_text())
            if (config["protocol"] != SCREENED_PROTOCOL or config["screened_pillar"] is not True
                    or config["render_distance_chunks"] != 4 or config["setup_revision"] != SETUP_REVISION
                    or config["world_seed"] != worlds[seed] or config["scenario_seed"] != int(seed)
                    or config["distance"] != distance
                    or config["return_limit"] != 2 * config["teacher_return_steps"] + 200
                    or config.get("outbound_error", "") != row["outbound_error"]):
                raise ValueError(f"Invalid screened fixture {distance}/{seed}")
            fixtures[f"{distance}/{seed}"] = config
    return dict(protocol=SCREENED_PROTOCOL, formal=frozen["formal"], fixtures=fixtures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--seeds-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = freeze(args.scores, args.seeds_file)
    with args.output.open("x") as stream:
        json.dump(frozen, stream)
        stream.write("\n")
    print(f"Frozen {len(frozen['fixtures'])} reviewed screened fixtures")


if __name__ == "__main__":
    main()
