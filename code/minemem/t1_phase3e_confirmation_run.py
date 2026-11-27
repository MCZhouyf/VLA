"""Schedule the predeclared confirmation and summarize its score and action records."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from phase1_metrics import wilson


ARMS = ("visual", "shuffle_visual")
DISTANCES = (30, 10, 60, 100)


def save(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def verify(run):
    subprocess.run(["sha256sum","--check","--quiet",str(run/"frozen-code.sha256")],
                   cwd=run/"frozen-source",check=True)
    for directory,manifest in ((run, "seed-plan.sha256"),(run,"harness.sha256"),(run/"frozen", "SHA256SUMS")):
        subprocess.run(["sha256sum","--check","--quiet",manifest],cwd=directory,check=True)


def run_job(run, group, gpu, seeds, common, distance=None, *, repaired_scene=False):
    frozen=run/"frozen-source/code/minemem/t1_exec.sh"
    target=run/"eval"/group
    target.mkdir(parents=True)
    result={}
    with TemporaryDirectory(prefix=f"minemem-t1-confirm-{gpu}-",dir="/dev/shm") as local:
        env=dict(os.environ,T1_LOCAL_DATABASE_ROOT=local)
        for attempt in (1,2):
            pending=[seed for seed in seeds if seed not in result]
            folder=target/f"attempt-{attempt}"
            harness=["bash",str(run/"harness/t1_phase3e_confirmation_gpu.sh"),str(run),str(gpu),str(folder)]
            if repaired_scene:
                harness.append("--repaired-scene")
            if distance is not None:
                command=([*harness,"--mode","a"] if repaired_scene else
                         ["bash",str(frozen),"waypoint-predictions",str(gpu),str(folder)])
                command.extend([*common,"--grid-plan",group.rsplit("/",1)[-1],"--distances",str(distance),
                                "--metrics",str(run/"frozen/metrics.json"),"--tracking-repair","--variant","spatial"])
            else:
                command=[*harness,*common,"--mode",group.split("/")[-1].split("-")[0]]
            command+=["--seeds",*map(str,pending)]
            with (target/f"attempt-{attempt}.log").open("x") as stream:
                for reservation in range(20):
                    status=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
                    if status!=75 or reservation==19:
                        break
                    sleep(30)
            partial={}
            table=folder/"episodes.csv"
            if table.exists():
                with table.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        seed=int(row["seed"])
                        if seed not in pending or row["success"] not in ("True","False"):
                            raise ValueError(f"Unexpected result for {group}: {row}")
                        if distance is not None:
                            row.update(arm=group.rsplit("/",1)[-1],material_slot=row["goal_slot"])
                        partial.setdefault(seed,[]).append(dict(row,source=str(folder)))
            twin=group.startswith("C/")
            for seed,rows in partial.items():
                if twin:
                    coverage={(row["material_slot"],row["arm"]) for row in rows}
                    if len(rows)==4 and coverage=={(s,a) for s in ("0","1") for a in ARMS}:
                        result[seed]=rows
                    elif len(rows)>4 or len(coverage)!=len(rows):
                        raise ValueError(f"Repeated material arm for {group}/{seed}")
                elif len(rows)==1:
                    result[seed]=rows
                else:
                    raise ValueError(f"Repeated outcome for {group}/{seed}")
            print(f"{group} gpu={gpu} attempt={attempt} status={status} complete={len(result)}/{len(seeds)}",flush=True)
            if set(result)==set(seeds):
                break
            if status in (0,75):
                raise RuntimeError(f"{group} ended with incomplete coverage and exit code {status}")
    if set(result)!=set(seeds):
        raise RuntimeError(f"Missing cases in {group}: {sorted(set(seeds)-result.keys())}")
    rows=[row for seed in seeds for row in result[seed]]
    with (target/"episodes.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def score(rows):
    wins=sum(row["success"]=="True" for row in rows)
    reasons=Counter(row["end_reason"] for row in rows)
    return dict(successes=wins,n=len(rows),wilson95=wilson(wins,len(rows)),
                room_correct=sum(row["predicted_slot"]==row["goal_slot"] for row in rows),
                distractor_first=reasons["distractor_first"],budget=reasons["budget"],
                deaths=sum(int(row["deaths"]) for row in rows), pose_jumps=sum(int(row["pose_jumps"]) for row in rows),
                stabilizer_steps=sum(int(row["stabilizer_steps"]) for row in rows),
                heading_guard_steps=sum(int(row["heading_guard_steps"]) for row in rows),
                waypoint_jumps_tail200=sum(int(row["waypoint_jumps_tail200"]) for row in rows))


def trace_stats(rows, distance):
    counts=Counter()
    footprint=[]
    for row in rows:
        root=Path(row["source"])
        tag=f"d{distance}-seed{row['seed']}"
        with (root/f"{tag}-actions.jsonl").open() as stream:
            actions=[json.loads(line) for line in stream]
        if len(actions)!=int(row["steps"]):
            raise ValueError(f"Action count differs from the score in {root}/{tag}")
        heading_hits=0
        for action in actions:
            proposed=action.get("proposed_environment",action)
            proposed_yaw,proposed_pitch=map(float,(proposed["yaw"],proposed["pitch"]))
            forward=bool(proposed["buttons"].get("forward"))
            turns=abs(proposed_yaw)>.01 or abs(proposed_pitch)>.01
            counts["forward" if forward else "turn" if turns else "other"]+=1
            counts["forward_and_camera"]+=int(forward and turns)
            counts["steps"]+=1
            if action.get("control_override")=="route_heading_guard":
                heading_hits+=1
                counts["guard_agrees"]+=int(abs(proposed_yaw-action["yaw"]) <= .01
                     and abs(proposed_pitch-action["pitch"]) <= .01 and proposed["buttons"]==action["buttons"])
        if heading_hits!=int(row["heading_guard_steps"]):
            raise ValueError(f"Heading guard tally differs from the score in {root}/{tag}")
        counts["heading_guard_steps"]+=heading_hits
        if distance==100:
            with gzip.open(root/f"{tag}-outbound.csv.gz","rt") as stream:
                cells={(math.floor(float(p["x"])),math.floor(float(p["z"]))) for p in csv.DictReader(stream)}
            xs,zs=zip(*cells)
            span_x,span_z=max(xs)-min(xs)+1,max(zs)-min(zs)+1
            footprint.append(dict(seed=int(row["seed"]),cells=len(cells),span_x=span_x,span_z=span_z,
                                  fits_within_128=max(span_x,span_z)<=128,discarded_cells=0))
    keys=("steps","forward","turn","other","forward_and_camera","heading_guard_steps","guard_agrees")
    return {key:counts[key] for key in keys},footprint


def twin_summary(rows, plan):
    by_seed={}
    for row in rows:
        by_seed.setdefault(int(row["seed"]),{})[row["arm"],int(row["material_slot"])]=row
    pairs=[]
    for seed in plan["twin_seeds"]:
        cells=by_seed[seed]
        entry=dict(seed=seed,arms={})
        for arm in ARMS:
            legs=[cells[arm,slot] for slot in (0,1)]
            entry["arms"][arm]=dict(
                predicted_slots=[int(row["predicted_slot"]) for row in legs],
                success=[row["success"]=="True" for row in legs],reasons=[row["end_reason"] for row in legs],
                flip=legs[0]["predicted_slot"]!=legs[1]["predicted_slot"])
        pairs.append(entry)
    return dict(pairs=pairs,summary={arm:dict(pairs=len(pairs),
                both_success=sum(all(r["arms"][arm]["success"]) for r in pairs),
                flips=sum(r["arms"][arm]["flip"] for r in pairs)) for arm in ARMS})


def run_suite(args):
    run=args.run
    plan=json.loads((run/"seed-plan.json").read_text())
    repaired_scene="scene_revision" in plan
    if not json.loads((run/"frozen/metrics.json").read_text())["passed"]:
        raise RuntimeError("The frozen CPU acceptance gate rejected these predeclared scenes")
    verify(run)
    by_distance={d:[row["seed"] for row in plan["cases"] if row["distance"]==d] for d in DISTANCES}
    weights=args.offline/"room-head.pt"
    offline=args.offline/"offline_gate.json"
    gate=json.loads(offline.read_text())
    if not gate["passed"] or hashlib.sha256(weights.read_bytes()).hexdigest()!=gate["weights_sha256"]:
        raise ValueError("Confirmation room weights do not match the frozen offline record")
    shared=["--training-run",str(args.controller),"--room-weights",str(weights),
            "--fixtures",str(run/"frozen/fixtures.json"),"--batch","3"]
    standard=[*shared,"--oracle-gate",str(args.oracle_gate),"--offline-gate",str(offline)]
    result={}
    with ThreadPoolExecutor(max_workers=2) as workers:
        for distance in DISTANCES:
            (run/"gpu.progress").write_text(f"A-{distance}\n")
            jobs={arm:workers.submit(run_job,run,f"A/{distance}/{arm}",gpu,by_distance[distance],standard,distance,
                                     repaired_scene=repaired_scene)
                  for arm,gpu in zip(ARMS,(6,7),strict=True)}
            groups={}
            for arm,future in jobs.items():
                rows=future.result()
                actions,footprint=trace_stats(rows,distance)
                groups[arm]=dict(score(rows),actions=actions)
                if footprint:
                    groups[arm]["footprint"]=footprint
            visual=groups["visual"]["successes"]
            advantage=visual-groups["shuffle_visual"]["successes"]
            result[distance]=dict(groups=groups,advantage=advantage,passed=visual>=23 and advantage>=6)
            save(run/f"A-{distance}.json",result[distance])
            verify(run)

        (run/"gpu.progress").write_text("B\n")
        shared_seeds=by_distance[30]
        ablations={}
        for first, second in (("b1","b2"),("b3",None)):
            jobs={first:workers.submit(run_job,run,f"B/{first}",6,shared_seeds,shared,repaired_scene=repaired_scene)}
            if second:
                jobs[second]=workers.submit(run_job,run,f"B/{second}",7,shared_seeds,shared,repaired_scene=repaired_scene)
            for name,future in jobs.items():
                ablations[name]=score(future.result())
                save(run/f"B-{name}.json",ablations[name])
        verify(run)
        (run/"gpu.progress").write_text("C\n")
        pairs=plan["twin_seeds"]
        jobs=[workers.submit(run_job,run,f"C/twin-{gpu}",gpu,pairs[index::2],shared,repaired_scene=repaired_scene)
              for index,gpu in enumerate((6,7))]
        twins=twin_summary([r for job in jobs for r in job.result()],plan)
    verify(run)
    save(run/"confirmation-decision.json",dict(frozen_commit=plan["frozen_commit"],
         scene_revision=plan.get("scene_revision","original"),A=result,B=ablations,C=twins,
         main_passed=result[30]["passed"]))
    (run/"gpu.progress").write_text("complete\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","controller","offline","oracle-gate"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    run_suite(parser.parse_args())
