"""Confirm Plan L with a learned goal, matched grid routing and material swaps."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from t1_phase3e_confirmation_run import run_job, save_or_verify, score, twin_summary, verify


def confirm(args):
    run = args.run
    verify(run)
    for manifest in ("cpu-harness.sha256","eval-harness.sha256"):
        subprocess.run(["sha256sum","--check","--quiet",manifest],cwd=run,check=True)
    plan = json.loads((run/"seed-plan.json").read_text())
    if plan["evaluation"]!="learned_cell_pointers" or not json.loads((run/"frozen/metrics.json").read_text())["passed"]:
        raise ValueError("The new 30-block cohort needs its original v3 teacher qualification")
    seeds = [row["seed"] for row in plan["cases"]]
    if len(seeds)!=30 or len(set(seeds))!=30 or len(set(plan["twin_seeds"]))!=10:
        raise ValueError("Expected a registered 30-world cohort and 10 material pairs")
    common = ["--training-run",str(args.controller),"--room-weights",str(args.room_weights),
              "--pointer-weights",str(run/"fit/pointer.pt"),"--fixtures",str(run/"frozen/fixtures.json"),
              "--batch","3"]
    harness = run/"eval-harness"
    with ThreadPoolExecutor(max_workers=2) as workers:
        (run/"gpu.progress").write_text("learned-and-grid\n")
        jobs = {name:workers.submit(run_job,run,f"L/{name}",gpu,seeds,[*common,"--pointer-type",name],
                                    recovery_harness=harness,harness_mode="pointer")
                for gpu,name in ((6,"learned"),(7,"grid"))}
        arms = {name:score(job.result()) for name,job in jobs.items()}
        save_or_verify(run/"pointer-closed-loop.json",arms)
        (run/"gpu.progress").write_text("material-swaps\n")
        jobs = [workers.submit(run_job,run,f"C/twin-{gpu}",gpu,plan["twin_seeds"][i::2],
                               [*common,"--pointer-type","learned"],recovery_harness=harness,
                               harness_mode="pointer-twin") for i,gpu in enumerate((6,7))]
        paired = twin_summary([row for job in jobs for row in job.result()],plan)
    summary = paired["summary"]["visual"]
    save_or_verify(run/"pointer-confirmation.json",dict(arms=arms,paired=paired,
                   passed=arms["learned"]["successes"]>=27 and summary["flips"]>=9))
    verify(run)
    (run/"gpu.progress").write_text("complete\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run","controller","room-weights"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    confirm(parser.parse_args())
