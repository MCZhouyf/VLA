"""Validate and summarize the matched three-variant T1 comparison."""

import argparse
import csv
import json
from pathlib import Path

from phase1_metrics import wilson
from phase1_t1 import DISTANCES
from t1_occlusion import SCREENED_PROTOCOL
from t1_safety import SETUP_REVISION


VARIANTS = ("no_memory", "long", "spatial")


def read(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def validate_training(runs):
    pairs = [(json.loads((path / "training_config.json").read_text()),
              json.loads((path / "summary.json").read_text())) for path in runs]
    if {config["variant"] for config, _ in pairs} != set(VARIANTS) or len(pairs) != 3:
        raise ValueError("Provide one training run for each T1 variant")
    equal = ("dataset_sha256", "checkpoint", "seed", "epochs", "learning_rate",
             "accumulation", "quantized")
    base_config, base_result = pairs[0]
    for config, result in pairs:
        if (any(config[key] != base_config[key] for key in equal)
                or any(config.get(key) != base_config.get(key)
                       for key in ("resolution", "validate_every", "checkpoint_selection"))
                or result["stopped_at_budget"] or config["budget_seconds"]):
            raise ValueError("Training data, hyperparameters, and complete epochs must match")
        if any(result[key] != base_result[key] for key in ("supervised_examples", "updates")):
            raise ValueError("Training runs consumed different examples or steps")
    return {config["variant"]: str(path.resolve()) for path, (config, _) in zip(runs, pairs)}


def summarize(rows, seeds):
    grid = {(row["model"], int(row["d"]), int(row["seed"])) for row in rows}
    expected = {(variant, distance, seed) for variant in VARIANTS
                for distance in DISTANCES for seed in seeds}
    if grid != expected or len(grid) != len(rows):
        raise ValueError("Formal evaluation coverage differs from the frozen cohort")
    protocols = {row.get("scene_protocol", "t1_v4") for row in rows}
    if len(protocols) != 1 or not protocols <= {"t1_v4", SCREENED_PROTOCOL}:
        raise ValueError("All results must use the same frozen scene protocol")
    for row in rows:
        outbound_failure = row["end_reason"].startswith("outbound_")
        ok = (float(row["min_dist"]) <= 3 and int(row["deaths"]) == 0
              and row["pose_jump"] == "False" and not outbound_failure)
        if (row["mode"] != "formal" or row["setup_revision"] != SETUP_REVISION
                or row["success"] != str(ok)
                or not int(not outbound_failure) <= int(row["steps"]) <= int(row["return_limit"])):
            raise ValueError(f"Invalid formal row: {row}")
    summary = []
    for variant in VARIANTS:
        for distance in DISTANCES:
            group = [row for row in rows if row["model"] == variant and int(row["d"]) == distance]
            count = sum(row["success"] == "True" for row in group)
            low, high = wilson(count, len(group))
            summary.append(dict(model=variant, distance=distance, successes=count, total=len(group),
                                rate=count / len(group), ci95_low=low, ci95_high=high))
    return summary


def plot(output, summaries, references=()):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    references_by_model = (
        ("random", "Random", ":"), ("steve1", "STEVE-1", "--"), ("jarvis7b", "JARVIS-VLA-7B", "-"),
    )
    for model, label, style in references_by_model:
        cohort = sorted((r for r in references if r["model"] == model), key=lambda r: int(r["d"]))
        if not cohort:
            continue
        if [int(row["d"]) for row in cohort] != list(DISTANCES) or any(int(row["seeds"]) != 5 for row in cohort):
            raise ValueError(f"Expected the complete frozen pilot summary for {model}")
        axis.plot(DISTANCES, [float(row["success_rate"]) for row in cohort],
                  linestyle=style, color="0.55", marker=".", label=f"{label} (5-seed pilot)")
    for variant, label in zip(VARIANTS, ("No memory", "8-frame history", "Spatial memory"), strict=True):
        cohort = [row for row in summaries if row["model"] == variant]
        means = [row["rate"] for row in cohort]
        intervals = ([row["rate"] - row["ci95_low"] for row in cohort],
                     [row["ci95_high"] - row["rate"] for row in cohort])
        axis.errorbar(DISTANCES, means, yerr=intervals, marker="o", capsize=4, label=label)
    axis.set(xlabel="Distance from landmark (blocks)", ylabel="Return success rate",
             xticks=DISTANCES, ylim=(0, 1.02))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runs", nargs=3, required=True, type=Path)
    parser.add_argument("--episodes", nargs="+", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--zero-shot-summary", type=Path)
    args = parser.parse_args()
    runs = validate_training(args.runs)
    seeds = json.loads(args.frozen.read_text())["formal"]
    if len(set(seeds)) != 30:
        raise ValueError("Expected the 30 frozen formal seeds")
    rows = read(args.episodes)
    if any(row["training_run"] != runs[row["model"]] for row in rows):
        raise ValueError("Evaluation output does not match the supplied training runs")
    summaries = summarize(rows, seeds)
    protocol = "screened" if rows[0].get("scene_protocol") == SCREENED_PROTOCOL else "v4"
    prefix = f"cliff_T1_{protocol}_trained"
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in ((f"{prefix}.csv", rows), (f"{prefix}_summary.csv", summaries)):
        with (args.output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    plot(args.output / f"{prefix}.svg", summaries)
    if args.zero_shot_summary:
        plot(args.output / f"cliff_T1_{protocol}_comparison.svg", summaries, read([args.zero_shot_summary]))
    rates = {row["model"]: row["rate"] for row in summaries if row["distance"] == 30}
    print(json.dumps(dict(d30=rates, proceed=rates["no_memory"] < 0.20 and rates["spatial"] > 0.50)))


if __name__ == "__main__":
    main()
