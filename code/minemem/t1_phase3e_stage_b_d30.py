"""Run all five frozen 30-block arms after the paired offline gate passes."""

import argparse
import csv
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

from phase1_metrics import wilson
from t1_phase3e_recover_eval import extend_scores, read_scores


def gpu_job(command, log, environment):
    with log.open("a") as stream:
        while True:
            rc = subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT).returncode
            if rc!=75:
                return rc
            sleep(30)


def execute_arm(entry, run, common, args, variant, gpu, seeds, environment):
    expected={(30,seed) for seed in seeds}
    scores={}
    root=run/variant
    root.mkdir()
    gpu_env=dict(environment,T1_LOCAL_DATABASE_ROOT=tempfile.mkdtemp(
        dir="/dev/shm",prefix=f"minemem-t1-b-{variant}-"))
    for i in range(1,4):
        pending=sorted(seed for _,seed in expected-scores.keys())
        if not pending:
            break
        output=root/f"attempt-{i}"
        log=root/f"attempt-{i}.log"
        command=["bash",str(entry),"waypoint-predictions",str(gpu),str(output),*common,
                 "--distances","30","--variant",variant,"--seeds",*map(str,pending)]
        status=gpu_job(command,log,gpu_env)
        extend_scores(scores,read_scores(output,expected-scores.keys()))
        print(f"d30 {variant} GPU{gpu} attempt={i} status={status} coverage={len(scores)}/30",flush=True)
        if len(scores)<30:
            trace=log.read_text(errors="replace")
            if "T1 /execute if block validation failed" not in trace and "GLFW" not in trace:
                raise RuntimeError(f"Unresolved simulator or prediction failure in {log}")
    if set(scores)!=expected:
        raise ValueError(f"Incomplete frozen coverage for {variant}")
    rows=list(scores.values())
    with (root/"episodes.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def oracle_scores(path, seeds):
    with path.open() as stream:
        rows=[row for row in csv.DictReader(stream) if row["d"]=="30"]
    if len(rows)!=30 or {int(row["seed"]) for row in rows}!=set(seeds):
        raise ValueError("Invalid shared-stabilizer oracle coverage")
    for row in rows:
        name=f"d30-seed{row['seed']}-actions.jsonl"
        with (Path(row["source"])/name).open() as log:
            row["stabilizer_steps"]=sum("control_override" in json.loads(line) for line in log)
    return rows


def report(rows):
    n=len(rows)
    successes=sum(row["success"]=="True" for row in rows)
    stability=[int(row.get("stabilizer_steps",0)) for row in rows]
    return dict(successes=successes,n=n,rate=successes/n,ci95=wilson(successes,n),
                deaths=sum(int(row["deaths"]) for row in rows),pose_jumps=sum(int(row["pose_jumps"]) for row in rows),
                stabilizer_steps=sum(stability),stabilizer_episodes=sum(count>0 for count in stability))


def run(args):
    gate=json.loads((args.offline/"offline_gate.json").read_text())
    if not gate["passed"]:
        raise ValueError("The v3 material-swap readout has not passed")
    if not json.loads((args.oracle/"recheck_oracle_gate.json").read_text())["passed"]:
        raise ValueError("The shared-stabilizer oracle has not passed")
    frozen=args.formal/"frozen"
    seeds=json.loads((frozen/"fixtures.json").read_text())["formal"]
    if len(seeds)!=30 or len(set(seeds))!=30:
        raise ValueError("Expected 30 unique frozen T1 scenarios")
    predictions=args.output
    entry=Path(__file__).with_name("t1_exec.sh")
    environment=dict(os.environ,WORK=str(predictions.parents[2]),TZ="Asia/Shanghai")
    chosen=args.output/"waypoints"
    local=Path((args.feature_run/"heads-local-cache").read_text().strip())
    cache=local if local.is_dir() else args.feature_run/"cache"
    train=["bash",str(entry),"waypoint-heads","6","--plan",str(args.feature_run/"cache-plan.json"),
           "--cache",str(cache),"--initial",str(args.initial),"--output",str(chosen),"--condition-candidate"]
    if gpu_job(train,predictions/"training.log",environment):
        raise RuntimeError("Candidate-conditioned waypoint training failed")
    common=["--training-run",str(args.formal/"controller"),"--metrics",str(frozen/"metrics.json"),
            "--fixtures",str(frozen/"fixtures.json"),"--offline-gate",str(args.offline/"offline_gate.json"),
            "--oracle-gate",str(args.oracle/"recheck_oracle_gate.json"),"--initial",str(args.initial),
            "--room-weights",str(args.room_weights),"--waypoint-weights",str(chosen/"waypoint-heads.pt")]
    with (frozen/"acceptance.csv").open() as stream:
        replay=[row for row in csv.DictReader(stream) if row["distance"]=="30" and row["baseline"]=="reverse_route"]
    if len(replay)!=30 or {int(row["seed"]) for row in replay}!=set(seeds):
        raise ValueError("The replay reference does not cover the frozen seeds")
    scores={"oracle":oracle_scores(args.oracle/"episodes.csv",seeds),"replay":replay}
    with ThreadPoolExecutor(max_workers=2) as pool:
        def no_memory_then_long():
            return {name:execute_arm(entry,predictions,common,args,name,6,seeds,environment)
                    for name in ("no_memory","long")}
        short=pool.submit(no_memory_then_long)
        spatial=pool.submit(execute_arm,entry,predictions,common,args,"spatial",7,seeds,environment)
        scores["spatial"]=spatial.result()
        scores.update(short.result())
    groups={name:report(rows) for name,rows in scores.items()}
    wins={name:value["successes"] for name,value in groups.items()}
    passed=(wins["oracle"]>=24 and wins["no_memory"]<=18 and wins["spatial"]>=23
            and wins["spatial"]-wins["no_memory"]>=6)
    summary=dict(distance=30,groups=groups,continue_to_far_distances=passed,
                 uses_shared_stabilizer=["oracle","no_memory","long","spatial"])
    (predictions/"d30_decision.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("formal","oracle","offline","room-weights","feature-run","initial","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    status=1
    try:
        run(args)
        status=0
    finally:
        (args.output/"exit_status").write_text(f"{status}\n")
