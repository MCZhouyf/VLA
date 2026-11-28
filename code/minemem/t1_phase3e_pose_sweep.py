"""Run the 360 paired noise evaluations on the original confirmation fixtures."""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from t1_phase3e_confirmation_run import run_job, save_or_verify, score, verify


LEVELS = ((0.05, 0.5), (0.1, 1.0), (0.2, 2.0))
DISTANCES = (30, 10, 60, 100)


def sweep(args):
    run = args.run
    verify(run)
    plan = json.loads((run/"seed-plan.json").read_text())
    if "scene_revision" in plan or len(plan["cases"]) != 120:
        raise ValueError("Expected the original balanced confirmation cohort")
    seed_sets = {d:[r["seed"] for r in plan["cases"] if r["distance"]==d] for d in DISTANCES}
    offline = args.offline / "offline_gate.json"
    gate = run / "gpu-eligibility.json"
    common = ["--training-run",str(args.controller),"--room-weights",str(args.offline/"room-head.pt"),
              "--fixtures",str(run/"frozen/fixtures.json"),"--batch","3","--oracle-gate",str(args.oracle_gate),
              "--offline-gate",str(offline),"--metrics",str(gate)]
    schedule = [(distance,position,yaw) for distance in DISTANCES for position,yaw in LEVELS]

    def lane(gpu, jobs):
        for distance,position,yaw in jobs:
            name = f"sigma-{position:g}-{yaw:g}"
            rows = run_job(run,f"{name}/{distance}/visual",gpu,seed_sets[distance],common,distance,
                           ground_repair=distance==60,pose_noise=(position,yaw),recovery_harness=run/"harness")
            samples = position_squared = yaw_squared = 0
            for row in rows:
                tag = f"d{distance}-seed{row['seed']}-pose-noise.json"
                evidence = json.loads((Path(row["source"])/tag).read_text())
                samples += evidence["samples"]
                position_squared += evidence["horizontal_rmse"]**2 * evidence["samples"]
                yaw_squared += evidence["yaw_rmse"]**2 * evidence["samples"]
            result = dict(distance=distance,position_sigma=position,yaw_sigma=yaw,
                          horizontal_rmse=math.sqrt(position_squared/samples),
                          yaw_rmse=math.sqrt(yaw_squared/samples),**score(rows))
            save_or_verify(run/f"{name}-d{distance}.json",result)
            print(f"d={distance} noise={position:g}/{yaw:g} success={result['successes']}/30",flush=True)

    with ThreadPoolExecutor(max_workers=2) as workers:
        jobs = [workers.submit(lane,gpu,schedule[index::2]) for index,gpu in enumerate((6,7))]
        for job in jobs:
            job.result()
    verify(run)
    save_or_verify(run/"pose-results.json",[json.loads((run/f"sigma-{p:g}-{y:g}-d{d}.json").read_text())
                                          for d,p,y in schedule])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run","controller","offline","oracle-gate"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    sweep(parser.parse_args())
