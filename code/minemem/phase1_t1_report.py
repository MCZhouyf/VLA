"""Validate benchmark coverage and write T1 success curves with Wilson intervals."""

import argparse
import csv
from pathlib import Path

from phase1_metrics import read_rows, wilson


MODELS = ("random", "steve1", "jarvis7b", "vpt")
DISTANCES = (10, 30, 60, 100)


def validate_results(inputs, seeds, models=MODELS):
    rows = list(read_rows(inputs))
    expected = {(name, d, seed) for name in models for d in DISTANCES for seed in seeds}
    actual = {(row["model"], int(row["d"]), int(row["seed"])) for row in rows}
    if len(actual) != len(rows) or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Unexpected coverage. Missing {missing[:8]}. Extra {extra[:8]}.")
    for row in rows:
        distance = float(row["final_dist"])
        closest = float(row["min_dist"])
        step = int(row["steps"])
        limit = int(row["return_limit"])
        if not 0 <= closest <= distance or not 1 <= step <= limit:
            raise ValueError(f"Invalid distances or step counts in {row}")
        if row["success"] != str(closest <= 3):
            raise ValueError(f"Success flag does not match minimum distance in {row}")
    return sorted(rows, key=lambda row: (row["model"], int(row["d"]), int(row["seed"])))


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, models=MODELS):
    summary = []
    for model in models:
        for distance in DISTANCES:
            cohort = [row for row in rows if row["model"] == model and int(row["d"]) == distance]
            successes = sum(row["success"] == "True" for row in cohort)
            total = len(cohort)
            lower, upper = wilson(successes, total)
            summary.append(dict(model=model, d=distance, successes=successes, seeds=total,
                                success_rate=successes / total, ci95_low=lower, ci95_high=upper))
    return summary


def plot(path, summary, monochrome=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axis = plt.subplots(figsize=(7.0, 4.4), layout="constrained")
    for name, label, style in (("random", "Random", ":"), ("steve1", "STEVE-1", "--"),
                               ("jarvis7b", "JARVIS-VLA-7B", "-")):
        cohort = [row for row in summary if row["model"] == name]
        if not cohort:
            continue
        rates = [row["success_rate"] for row in cohort]
        lower = [rate - row["ci95_low"] for rate, row in zip(rates, cohort)]
        upper = [row["ci95_high"] - rate for rate, row in zip(rates, cohort)]
        appearance = dict(color="0.45", linestyle=style) if monochrome else {}
        axis.errorbar([row["d"] for row in cohort], rates, yerr=(lower, upper), marker="o", capsize=4,
                      label=label, **appearance)
    axis.set(xlabel="Distance from landmark (blocks)", ylabel="Return success rate",
             xticks=DISTANCES, ylim=(0, 1.02))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) not in (5, 30):
        parser.error("Provide the 5 or 30 distinct frozen scenario seeds")
    rows = validate_results(args.inputs, args.seeds)
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cliff_T1.csv", [row for row in rows if row["model"] != "vpt"])
    write_csv(args.output / "vpt_T1_appendix.csv", [row for row in rows if row["model"] == "vpt"])
    write_csv(args.output / "cliff_T1_summary.csv", summary)
    plot(args.output / "cliff_T1.svg", summary)


if __name__ == "__main__":
    main()
