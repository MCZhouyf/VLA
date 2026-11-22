"""Summarize T1 respawn reruns and render trajectory and success-rate figures."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from statistics import fmean

from phase1_metrics import wilson
from phase1_t1_report import plot, summarize, validate_results, write_csv
from t1_safety import SETUP_REVISION
from t1_v4 import DISTANCES, TEACHER_SEED


def acceptance_summary(directories):
    records = {}
    for directory in directories:
        for source in sorted(directory.glob("worker-*/acceptance.csv")):
            with source.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    key = int(row["d"]), int(row["seed"])
                    if key in records and not records[key]["error"]:
                        raise ValueError(f"Duplicate T1 acceptance result: {key}")
                    records[key] = row
    expected = {(d, seed) for d in DISTANCES for seed in range(TEACHER_SEED, TEACHER_SEED+50)}
    if not records.keys() <= expected:
        raise ValueError("T1 acceptance included an unexpected scenario or distance")
    summaries = []
    for d in DISTANCES:
        rows = [r for (distance, _), r in records.items()
                if distance == d and not r["error"] and r["setup_revision"] == SETUP_REVISION]
        n = len(rows)
        teachers = sum(r["teacher_success"] == "True" for r in rows)
        shortcuts = sum(r["shortcut_success"] == "True" for r in rows)
        for row in rows:
            for name in ("teacher", "shortcut"):
                if row[f"{name}_success"] == "True" and (
                   int(row[f"{name}_deaths"]) or row[f"{name}_pose_jump"] != "False"):
                    raise ValueError("Invalid death or pose jump on a successful trial")
        lower, upper = wilson(teachers, 50)
        teacher_death_episodes = sum(int(r["teacher_deaths"]) > 0 for r in rows)
        shortcut_death_episodes = sum(int(r["shortcut_deaths"]) > 0 for r in rows)
        summaries.append(dict(
            d=d, completed=n, teacher_success=teachers, teacher_ci95_low=lower,
            teacher_ci95_high=upper,
            mean_teacher_steps=fmean(int(r["teacher_steps"]) for r in rows
                                     if r["teacher_success"] == "True") if teachers else "",
            shortcut_success=shortcuts, teacher_death_episodes=teacher_death_episodes,
            teacher_deaths=sum(int(r["teacher_deaths"]) for r in rows),
            teacher_pose_jumps=sum(int(r["teacher_pose_jumps"]) for r in rows),
            shortcut_death_episodes=shortcut_death_episodes,
            shortcut_deaths=sum(int(r["shortcut_deaths"]) for r in rows),
            shortcut_pose_jumps=sum(int(r["shortcut_pose_jumps"]) for r in rows),
            accepted=n == 50 and teachers >= 48 and (d == 10 or shortcuts < 5),
        ))
    return summaries


def pilot_report(args):
    models = args.models
    seeds = json.loads(args.seeds_file.read_text())["pilot"]
    rows = validate_results(args.inputs, seeds, models)
    if any(row["setup_revision"] != SETUP_REVISION for row in rows):
        raise ValueError("Pilot rows must use the revised setup")
    summary = summarize(rows, models)
    for value in summary:
        group = [r for r in rows if r["model"] == value["model"] and int(r["d"]) == value["d"]]
        episodes = sum(int(r["deaths"]) > 0 for r in group)
        value.update(death_episodes=episodes, death_rate=episodes / len(group),
                     deaths=sum(int(r["deaths"]) for r in group),
                     pose_jump_episodes=sum(r["pose_jump"] == "True" for r in group),
                     pose_jumps=sum(int(r["pose_jumps"]) for r in group))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "pilot_episodes.csv", rows)
    write_csv(args.output / "pilot_summary.csv", summary)
    plot(args.output / "cliff_T1_respawn.svg", summary, monochrome=True)
    if "vpt" in models:
        vpt = [row for row in summary if row["model"] == "vpt"]
        plot(args.output / "vpt_respawn_appendix.svg", vpt, monochrome=True, include_vpt=True)


def read_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def trajectory(args):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    scene = json.loads(args.config.read_text())
    x0, z0 = scene["landmark"]
    outbound = read_trace(args.outbound)
    returned = read_trace(args.trace)
    if not outbound or not returned:
        raise ValueError("Trajectory plots require recorded outbound and return steps")
    figure, axis = plt.subplots(figsize=(7.6, 6.5), layout="constrained")

    def position(row):
        return float(row["x"])-x0, float(row["z"])-z0

    axis.plot(*zip(*(position(row) for row in outbound)), "--", color="#677480",
              linewidth=1.4, label="Teacher outbound")
    axis.plot(*zip(*(position(row) for row in returned)), color="#176f82", linewidth=1.55,
              label=f"{args.policy_label} return")
    axis.add_patch(Circle((0, 0), 3, facecolor="#edc461", edgecolor="#a87116",
                          alpha=.20, label="3-block success radius"))
    axis.scatter(0, 0, color="#ce8418", marker="*", s=160, zorder=5, label="Gold pillar")
    points = [(item["horizontal"][0]-x0, item["horizontal"][1]-z0)
              for item in scene["distractors"]]
    axis.scatter(*zip(*points), marker="s", color="#857495", s=32, label="Distractor pillars")
    for (x,z), item in zip(points, scene["distractors"], strict=True):
        axis.annotate(item["material"].replace("_block", ""), (x,z), xytext=(5,4),
                      textcoords="offset points", fontsize=8)
    if scene["wall"]:
        a, _, b, c, _, d = scene["wall"]
        axis.add_patch(Rectangle((a-x0,b-z0),c-a+1,d-b+1,color="#61636b",alpha=.4,label="Wall"))
    axis.scatter(*position(returned[0]), marker="^", color="#643e87", s=62, zorder=5,
                 label="Outbound endpoint")
    axis.scatter(*position(returned[-1]), marker="x", color="#b23d33", s=54, zorder=5,
                 label="Return end")
    jumps = [position(r) for r in returned if r.get("pose_jump") == "True"]
    if jumps:
        axis.scatter(*zip(*jumps), facecolors="none", edgecolors="#a13f39", s=95, label="Pose jump")
    axis.set(xlabel="x from gold pillar (blocks)", ylabel="z from gold pillar (blocks)",
             title=f"{args.policy_label} at {scene['distance']} blocks · seed {scene['scenario_seed']}")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(color="#d6dce0", linewidth=.65)
    axis.legend(loc="best", fontsize=8)
    figure.savefig(args.output)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    accept = sub.add_parser("acceptance")
    accept.add_argument("output", type=Path)
    accept.add_argument("inputs", type=Path, nargs="+")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("output", type=Path)
    pilot.add_argument("--inputs", type=Path, nargs="+", required=True)
    pilot.add_argument("--seeds-file", type=Path, required=True)
    pilot.add_argument("--models", nargs="+", default=("random","steve1","vpt"))
    trace = sub.add_parser("trajectory")
    trace.add_argument("output", type=Path)
    trace.add_argument("--config", type=Path, required=True)
    trace.add_argument("--outbound", type=Path, required=True)
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--policy-label", default="VPT-BC")
    args = parser.parse_args()
    if args.mode == "acceptance":
        rows = acceptance_summary(args.inputs)
        write_csv(args.output, rows)
        return 0 if all(row["accepted"] for row in rows) else 1
    if args.mode == "pilot":
        pilot_report(args)
    else:
        trajectory(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
