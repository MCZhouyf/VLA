"""Split the final long-window arm into the same fixed three-scene rollout lanes."""

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

from t1_phase3e_recover_eval import extend_scores, read_scores


def score_shard(launcher, output, shared, seeds, gpu):
    path = output/f"gpu{gpu}"
    runner = ["bash",str(launcher),"waypoint-predictions",str(gpu),str(path),*shared,
              "--seeds",*map(str,seeds)]
    environment = dict(os.environ,T1_PHASE3E_LONG_CHILD="1")
    log = output/f"gpu{gpu}.log"
    with log.open("x") as stream:
        while True:
            status = subprocess.run(runner,env=environment,stdout=stream,stderr=subprocess.STDOUT).returncode
            if status!=75:
                break
            sleep(20)
    if status:
        print(f"GPU {gpu} long-window retry evidence",flush=True)
        print("\n".join(log.read_text(errors="replace").splitlines()[-32:]),flush=True)
    return status,read_scores(path,{(30,seed) for seed in seeds})


def run(output, invocation):
    marker=invocation.index("--seeds")
    shared=invocation[:marker]
    seeds=list(map(int,invocation[marker+1:]))
    if (shared[shared.index("--variant")+1]!="long" or shared[shared.index("--distances")+1]!="30"
            or not seeds or len(set(seeds))!=len(seeds)):
        raise ValueError("The two-card scheduler is only scoped to the frozen 30-block long-window arm")
    midpoint=3*((len(seeds)+5)//6)
    shards={6:seeds[:midpoint],7:seeds[midpoint:]}
    output.mkdir()
    (output/"parallel_layout.json").write_text(json.dumps(shards,indent=2)+"\n")
    launcher=Path(__file__).with_name("t1_exec.sh")
    scores={}
    status=0
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(score_shard,launcher,output,shared,part,gpu) for gpu,part in shards.items() if part]
        for job in jobs:
            failed,rows=job.result()
            extend_scores(scores,rows)
            status|=int(failed!=0)
    if scores:
        fields=[key for key in next(iter(scores.values())) if key!="source"]
        with (output/"episodes.csv").open("x",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
            writer.writeheader()
            for (_,seed),row in sorted(scores.items()):
                source=Path(row["source"])
                for log in source.glob(f"d30-seed{seed}*"):
                    if log.is_file():
                        os.link(log,output/log.name)
                writer.writerow(row)
    return status if len(scores)==len(seeds) else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("invocation",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    raise SystemExit(run(args.output,args.invocation))
