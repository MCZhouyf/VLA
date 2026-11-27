"""Run and verify the two frozen 30-block grid planner arms on GPUs 6 and 7."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from t1_phase3e_closed_audit import large_turn_switches
from t1_phase3e_grid_plan import TRACKING_REPAIR
from t1_phase3e_recover_eval import extend_scores, read_scores
from t1_phase3e_stage_b_d30 import report


ARMS = {"visual": 6, "shuffle_visual": 7}


def run_arm(output, arm, gpu, common, expected):
    launcher = Path(__file__).with_name("t1_exec.sh")
    output.mkdir()
    scores = {}
    with TemporaryDirectory(prefix=f"minemem-t1-p-{gpu}-", dir="/dev/shm") as database:
        env = dict(os.environ, T1_LOCAL_DATABASE_ROOT=database)
        for attempt in (1, 2):
            pending = expected - scores.keys()
            command = [
                "bash", str(launcher), "waypoint-predictions", str(gpu), str(output / f"attempt-{attempt}"),
                *common, "--grid-plan", arm, "--seeds", *[str(seed) for _, seed in sorted(pending)],
            ]
            with (output / f"attempt-{attempt}.log").open("x") as stream:
                for lock_try in range(10):
                    status = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
                    if status != 75 or lock_try == 9:
                        break
                    sleep(30)
            additional = read_scores(output / f"attempt-{attempt}", pending)
            extend_scores(scores, additional)
            print(f"{arm} gpu={gpu} attempt={attempt} status={status} scored={len(scores)}/30", flush=True)
            if scores.keys() == expected:
                break
            if status == 0 or status == 75:
                raise RuntimeError(f"{arm} stopped with status {status} and {len(scores)}/30 scores")
    if scores.keys() != expected:
        raise RuntimeError(f"{arm} incomplete after retry; see {output}/attempt-*.log")
    rows = [scores[key] for key in sorted(expected)]
    with (output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def arm_summary(rows):
    switches = []
    guard_steps = []
    for row in rows:
        path = Path(row["source"]) / f"d30-seed{row['seed']}-actions.jsonl"
        controls = [json.loads(line) for line in path.read_text().splitlines()]
        count = large_turn_switches(controls)
        stabilized = sum(item.get("control_override")=="waypoint_turn_stabilizer" for item in controls)
        guards = [item for item in controls if item.get("control_override")=="route_heading_guard"]
        if (len(controls) != int(row["steps"]) or count != int(row["waypoint_jumps_tail200"])
                or stabilized != int(row["stabilizer_steps"]) or len(guards) != int(row.get("heading_guard_steps",0))
                or any(item["buttons"]["forward"] or not 0 < abs(item["yaw"]) <= 10 for item in guards)):
            raise ValueError(f"Action trace does not agree with the score for {path}")
        switches.append(count)
        guard_steps.append(len(guards))
    result = dict(report(rows), room_correct=sum(row["predicted_slot"] == row["goal_slot"] for row in rows),
                  waypoint_jumps_tail200=sum(switches), waypoint_jump_episodes=sum(count > 0 for count in switches))
    if "heading_guard_steps" in rows[0]:
        result.update(heading_guard_steps=sum(guard_steps), heading_guard_episodes=sum(n > 0 for n in guard_steps))
    return result


def run(args):
    frozen = args.formal / "frozen"
    fixtures = json.loads((frozen / "fixtures.json").read_text())
    seeds = fixtures["formal"]
    expected = {(30, seed) for seed in seeds}
    if len(seeds) != 30 or len(expected) != 30:
        raise ValueError("Grid planning requires 30 distinct frozen seeds")
    weights = args.offline / "room-head.pt"
    offline = args.offline / "offline_gate.json"
    gate = json.loads(offline.read_text())
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    if not (gate["primary_distance"] == 30 and gate["passed"] and gate["weights_sha256"] == weight_hash):
        raise ValueError("Grid planning requires the exact readout that passed the no-anchor gate")
    if not json.loads(args.oracle_gate.read_text())["passed"]:
        raise ValueError("The shared controller oracle gate did not pass")
    common = [
        "--training-run", str(args.formal / "controller"), "--metrics", str(frozen / "metrics.json"),
        "--fixtures", str(frozen / "fixtures.json"), "--offline-gate", str(offline),
        "--oracle-gate", str(args.oracle_gate), "--room-weights", str(weights),
        "--variant", "spatial", "--distances", "30", "--batch", "3",
    ]
    if args.tracking_repair:
        common.append("--tracking-repair")
    args.output.mkdir()
    setup = dict(gpus=ARMS, seeds=seeds, batch=3, room_weights_sha256=weight_hash, arguments=common)
    if args.tracking_repair:
        setup["tracking_repair"] = TRACKING_REPAIR
    (args.output / "launch.json").write_text(json.dumps(setup, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
        jobs = {arm: pool.submit(run_arm, args.output / arm, arm, gpu, common, expected)
                for arm, gpu in ARMS.items()}
        arms = {arm: arm_summary(job.result()) for arm, job in jobs.items()}
    visual = arms["visual"]["successes"]
    difference = visual - arms["shuffle_visual"]["successes"]
    decision = dict(arms=arms, offline_gate_passed=True, success_threshold=23, gap_threshold=6,
                    visual_success_gap=difference, passed=visual >= 23 and difference >= 6)
    if args.tracking_repair:
        decision["tracking_repair"] = TRACKING_REPAIR
    (args.output / "d30_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("formal", "oracle-gate", "offline", "output"):
        parser.add_argument(f"--{item}", type=Path, required=True)
    parser.add_argument("--tracking-repair", action="store_true")
    run(parser.parse_args())
