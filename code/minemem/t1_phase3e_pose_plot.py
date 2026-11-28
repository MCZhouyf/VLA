"""Draw the paired T1 success-versus-pose-error curve with Wilson intervals."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(confirmation, noise, output):
    audited = json.loads(confirmation.read_text())["A"]
    measurements = json.loads(noise.read_text())
    by_distance = {}
    rows = []
    for distance in (10,30,60,100):
        zero = dict(audited[str(distance)]["groups"]["visual"],position_sigma=0.,yaw_sigma=0.,
                    horizontal_rmse=0.,yaw_rmse=0.)
        series = [zero,*sorted((r for r in measurements if r["distance"]==distance),key=lambda r:r["position_sigma"])]
        if [r["position_sigma"] for r in series]!=[0.,.05,.1,.2] or any(r["n"]!=30 for r in series):
            raise ValueError(f"Expected all 120 paired outcomes at distance {distance}")
        by_distance[distance] = series
        for row in series:
            bounds = row["wilson95"]
            keys = ("position_sigma","yaw_sigma","horizontal_rmse","yaw_rmse","successes","n")
            rows.append(dict(distance=distance,**{key:row[key] for key in keys},lo=bounds[0],hi=bounds[1]))
    figure,axis = plt.subplots(figsize=(6.7,4.0),layout="constrained")
    for distance,series in by_distance.items():
        x = [r["horizontal_rmse"] for r in series]
        y = [r["successes"]/r["n"] for r in series]
        line, = axis.plot(x,y,"o-",label=f"{distance} blocks")
        low,high = zip(*(r["wilson95"] for r in series))
        axis.fill_between(x,low,high,color=line.get_color(),alpha=.11)
    axis.set(xlabel="Measured horizontal pose RMSE (blocks)",ylabel="Success rate",ylim=(0,1.04))
    axis.set_xticks((0,.071,.141,.283),("0","0.07","0.14","0.28"))
    heading = axis.secondary_xaxis("top")
    heading.set_xticks((0,.071,.141,.283),("0","0.5","1","2"))
    heading.set_xlabel("Paired yaw noise σ (degrees)")
    axis.grid(alpha=.2)
    axis.legend(loc="lower left",ncols=2,frameon=False)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    with output.with_suffix(".csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation","noise","output"):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    plot(args.confirmation,args.noise,args.output)
