"""Draw the measured formal validation curves and mark saved checkpoints."""

import argparse
import csv
import io
import json
from pathlib import Path
from time import sleep

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


LABELS = {"no_memory":"No memory", "long":"8-frame history", "spatial":"Spatial memory"}


def collection(full):
    for variant in LABELS:
        run = full/f"train-{variant}"
        path = run/"val_loss.csv"
        if path.is_file():
            yield variant, path.read_bytes()


def plot(data, output):
    rows_by_model = {model: list(csv.DictReader(io.StringIO(source.decode())))
                     for model,source in data}
    rows_by_model = {model: rows for model,rows in rows_by_model.items()
                     if rows and all(row.get("token_accuracy") for row in rows)}
    if not rows_by_model:
        return
    figure, (loss, accuracy) = plt.subplots(2,1,figsize=(7.2,6),sharex=True,layout="constrained")
    for model,rows in rows_by_model.items():
        updates=[int(r["update"]) for r in rows]
        label=LABELS[model]
        loss.plot(updates,[float(r["loss"]) for r in rows],marker="o",label=label)
        accuracy.plot(updates,[float(r["token_accuracy"]) for r in rows],marker="o",label=label)
    loss.set_ylabel("Action-token cross-entropy")
    accuracy.set_ylabel("Action-token accuracy")
    accuracy.set_xlabel("Optimizer step")
    accuracy.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for axis in (loss,accuracy):
        axis.grid(alpha=.25)
    loss.legend()
    temporary=output.with_suffix(".svg.tmp")
    figure.savefig(temporary,format="svg")
    plt.close(figure)
    temporary.replace(output)


def completed(full, data):
    if len(data) != len(LABELS):
        return False
    for model,source in data:
        summary = full/f"train-{model}"/"summary.json"
        rows=list(csv.DictReader(io.StringIO(source.decode())))
        if not summary.is_file() or not rows or str(json.loads(summary.read_text())["updates"])!=rows[-1]["update"]:
            return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--full",type=Path,required=True)
    parser.add_argument("--watch",action="store_true")
    args=parser.parse_args()
    last=None
    while True:
        data=list(collection(args.full))
        if data!=last:
            plot(data,args.output)
            last=data
        if not args.watch or completed(args.full,data):
            return
        sleep(45)


if __name__=="__main__":
    main()
