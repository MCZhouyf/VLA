"""Audit learned waypoint rollouts without changing the frozen evaluation."""

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

from t1_waypoint import RouteLookahead


WAYPOINT = re.compile(r"waypoint: (ahead|left|right) (\d+) deg, (\d+) blocks")
ARMS = ("no_memory","long","spatial")


def waypoint(prompt):
    side,angle,distance = WAYPOINT.search(prompt).groups()
    return int(angle)*(-1 if side=="left" else 1),int(distance)


def summarize_episode(row):
    root = Path(row["source"])
    tag = f"d{row['d']}-seed{row['seed']}"
    config = json.loads((root/f"{tag}.json").read_text())
    with gzip.open(root/f"{tag}-return.csv.gz","rt") as stream:
        states = list(csv.DictReader(stream))
    controls = [json.loads(line) for line in (root/f"{tag}-actions.jsonl").read_text().splitlines()]
    if len(controls)!=int(row["steps"]) or len(states)!=len(controls)+1 or config["goal_slot"]!=int(row["goal_slot"]):
        raise ValueError(f"Inconsistent frozen rollout data for {root}/{tag}")
    start = config["return_pose"]
    route = RouteLookahead((start["x"],start["z"]),config["teacher_targets"])
    diagnostics = []
    for control,state in zip(controls,states[:-1],strict=True):
        pose=[float(state[k]) for k in ("x","z","yaw")]
        learn=waypoint(control["instruction"])
        target=waypoint(route.condition(pose))
        error=abs((learn[0]-target[0]+180)%360-180)
        diagnostics.append(dict(predicted=learn,target=target,angle_error=error))
    tail = controls[-200:]
    tail_states = states[-len(tail)-1:]
    positions = [(float(state["x"]),float(state["z"])) for state in tail_states]
    travel = sum(math.dist(a,b) for a,b in zip(positions,positions[1:]))
    motion = [float(item["yaw"]) for item in tail]
    turns = sum(value!=0 for value in motion)
    alternations = sum(a*b<0 for a,b in zip(motion,motion[1:]))
    forward = sum(bool(item["buttons"].get("forward")) for item in tail)
    repeated_turn = (len(tail)==200 and alternations>=190 and turns>=190 and travel<.20 and not forward)
    overrides = sum("control_override" in item for item in controls)
    if overrides!=int(row["stabilizer_steps"]):
        raise ValueError(f"Stabilizer count differs for {tag}")
    return dict(seed=int(row["seed"]),d=int(row["d"]),steps=len(controls),success=row["success"]=="True",
                reason=row["end_reason"],room_correct=row["predicted_slot"]==row["goal_slot"],
                first=diagnostics[0],min_dist=float(row["min_dist"]),stabilizer_steps=overrides,
                deaths=int(row["deaths"]),pose_jumps=int(row["pose_jumps"]),
                tail_path=travel,tail_turns=turns,tail_alternations=alternations,tail_forward=forward,
                tail_stationary_alternation=repeated_turn,
                tail_most_common_prompts=Counter(item["predicted"] for item in diagnostics[-len(tail):]).most_common(5))


def read_arm(root):
    rows = []
    for path in sorted(root.glob("attempt-*/episodes.csv")):
        with path.open() as stream:
            rows.extend(dict(row,source=str(path.parent)) for row in csv.DictReader(stream))
    if len({(row["d"],row["seed"]) for row in rows})!=len(rows):
        raise ValueError(f"Repeated scores in {root}")
    return [summarize_episode(row) for row in rows]


def audit(root):
    overview = {}
    for variant in ARMS:
        rows = read_arm(root/variant)
        overview[variant] = dict(n=len(rows),wins=sum(row["success"] for row in rows),
                                 room_correct=sum(row["room_correct"] for row in rows),
                                 wins_with_correct_room=sum(row["success"] and row["room_correct"] for row in rows),
                                 stationary_alternation=sum(row["tail_stationary_alternation"] for row in rows),
                                 reasons=dict(Counter(row["reason"] for row in rows)),episodes=rows)
    return overview


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=audit(args.run)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    else:
        print(json.dumps(result,indent=2))
