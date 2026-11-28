"""Schedule the predeclared confirmation and summarize its score and action records."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from phase1_metrics import wilson


ARMS = ("visual", "shuffle_visual")
DISTANCES = (30, 10, 60, 100)
SCRIPT_BASELINES = ("turn_180", "scan", "find_structure", "start_pose")
GRAVEL_WORLD = 88407245


def save(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def verify(run):
    subprocess.run(["sha256sum","--check","--quiet",str(run/"frozen-code.sha256")],
                   cwd=run/"frozen-source",check=True)
    for directory,manifest in ((run, "seed-plan.sha256"),(run,"harness.sha256"),(run/"frozen", "SHA256SUMS")):
        subprocess.run(["sha256sum","--check","--quiet",manifest],cwd=directory,check=True)


def attempt_rows(folder, group, seeds, distance):
    partial={}
    table=folder/"episodes.csv"
    if table.exists():
        with table.open(newline="") as stream:
            for row in csv.DictReader(stream):
                seed=int(row["seed"])
                if seed not in seeds or row["success"] not in ("True","False"):
                    raise ValueError(f"Unexpected result for {group}: {row}")
                if distance is not None:
                    row.update(arm=group.rsplit("/",1)[-1],material_slot=row["goal_slot"])
                partial.setdefault(seed,[]).append(dict(row,source=str(folder)))
    complete={}
    for seed,rows in partial.items():
        if group.startswith("C/"):
            coverage={(row["material_slot"],row["arm"]) for row in rows}
            if len(rows)==4 and coverage=={(s,a) for s in ("0","1") for a in ARMS}:
                complete[seed]=rows
            elif len(rows)>4 or len(coverage)!=len(rows):
                raise ValueError(f"Repeated material arm for {group}/{seed}")
        elif len(rows)==1:
            complete[seed]=rows
        else:
            raise ValueError(f"Repeated outcome for {group}/{seed}")
    return complete


def run_job(run, group, gpu, seeds, common, distance=None, *, repaired_scene=False, ground_repair=False,
            recovery_harness=None):
    frozen=run/"frozen-source/code/minemem/t1_exec.sh"
    target=run/"eval"/group
    target.mkdir(parents=True,exist_ok=recovery_harness is not None)
    result={}
    first_attempt=1
    if recovery_harness:
        attempts=sorted(int(path.stem.split("-")[-1]) for path in target.glob("attempt-*.log"))
        for attempt in attempts:
            pending=set(seeds)-result.keys()
            result.update(attempt_rows(target/f"attempt-{attempt}",group,pending,distance))
        if attempts:
            first_attempt=attempts[-1]+1
    harness_dir=recovery_harness or run/"harness"
    with TemporaryDirectory(prefix=f"minemem-t1-confirm-{gpu}-",dir="/dev/shm") as local:
        env=dict(os.environ,T1_LOCAL_DATABASE_ROOT=local)
        for attempt in range(first_attempt,first_attempt+2):
            pending=[seed for seed in seeds if seed not in result]
            if not pending:
                break
            folder=target/f"attempt-{attempt}"
            harness=["bash",str(harness_dir/"t1_phase3e_confirmation_gpu.sh"),str(run),str(gpu),str(folder)]
            if repaired_scene:
                harness.append("--repaired-scene")
            if ground_repair:
                harness.extend(("--ground-repair-world",str(GRAVEL_WORLD)))
            if distance is not None:
                command=([*harness,"--mode","a"] if repaired_scene or ground_repair else
                         ["bash",str(frozen),"waypoint-predictions",str(gpu),str(folder)])
                command.extend([*common,"--grid-plan",group.rsplit("/",1)[-1],"--distances",str(distance),
                                "--tracking-repair","--variant","spatial"])
            else:
                command=[*harness,*common,"--mode",group.split("/")[-1].split("-")[0]]
            command+=["--seeds",*map(str,pending)]
            with (target/f"attempt-{attempt}.log").open("x") as stream:
                for reservation in range(20):
                    status=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
                    if status!=75 or reservation==19:
                        break
                    sleep(30)
            result.update(attempt_rows(folder,group,pending,distance))
            print(f"{group} gpu={gpu} attempt={attempt} status={status} complete={len(result)}/{len(seeds)}",flush=True)
            if set(result)==set(seeds):
                break
            if status in (0,75):
                raise RuntimeError(f"{group} ended with incomplete coverage and exit code {status}")
    if set(result)!=set(seeds):
        raise RuntimeError(f"Missing cases in {group}: {sorted(set(seeds)-result.keys())}")
    rows=[row for seed in seeds for row in result[seed]]
    table=target/"episodes.csv"
    if table.exists():
        with table.open(newline="") as stream:
            if list(csv.DictReader(stream))!=rows:
                raise ValueError(f"Previously summarized rows differ for {group}")
    else:
        with table.open("x",newline="") as stream:
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


def evaluation_inputs(run, protocol_update):
    verify(run)
    plan=json.loads((run/"seed-plan.json").read_text())
    metrics_path=run/"frozen/metrics.json"
    metrics=json.loads(metrics_path.read_text())
    if not protocol_update:
        if not metrics["passed"]:
            raise RuntimeError("The frozen CPU acceptance gate rejected these predeclared scenes")
        return plan,metrics_path
    cases=plan["cases"]
    if (metrics["missing"] or len(cases)!=120 or len({row["seed"] for row in cases})!=120
            or any(group["n"]!=30 for table in metrics["distances"].values() for group in table.values())
            or any(metrics["distances"][str(d)]["teacher"]["rate"]<.95 for d in DISTANCES)):
        raise ValueError("The confirmation requires complete CPU coverage and validated teacher routes")
    eligibility=run/"gpu-eligibility.json"
    record=dict(passed=True,protocol="2026-11-27_confirmation_update",legacy_cpu_passed=metrics["passed"],
                script_scores_used_for="comparison",cpu_metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest())
    save_or_verify(eligibility,record)
    return plan,eligibility


def save_or_verify(path, data):
    if not path.exists():
        save(path,data)
    elif json.loads(path.read_text())!=json.loads(json.dumps(data)):
        raise ValueError(f"Existing summary differs from recalculated data: {path}")


def paired_difference(visual, baseline):
    """Percentile bootstrap over whole seed pairs, enumerated exactly."""
    if not visual or set(visual)!=set(baseline):
        raise ValueError("The comparison must contain the same seed pairs")
    deltas=[int(value)-int(baseline[seed]) for seed,value in visual.items()]
    counts=Counter(deltas)
    n=len(deltas)
    positive,negative,unchanged=(counts[value]/n for value in (1,-1,0))
    probabilities=defaultdict(float)
    for wins in range(n+1):
        for losses in range(n-wins+1):
            ties=n-wins-losses
            probabilities[wins-losses]+=(math.comb(n,wins)*math.comb(n-wins,losses)
                                        *positive**wins*negative**losses*unchanged**ties)
    cumulative=0.
    quantiles=[]
    for delta,probability in sorted(probabilities.items()):
        cumulative+=probability
        while len(quantiles)<2 and cumulative>=(.025,.975)[len(quantiles)]:
            quantiles.append(delta/n)
    return dict(gap=sum(deltas), n=n, visual_only=counts[1], baseline_only=counts[-1],
                ties=counts[0], bootstrap95=quantiles, method="paired_percentile_bootstrap_exact_enumeration")


def main_decision(run, groups, visual_rows):
    with (run/"frozen/acceptance.csv").open(newline="") as stream:
        scripts={name:{} for name in SCRIPT_BASELINES}
        for row in csv.DictReader(stream):
            if row["distance"]=="30" and row["baseline"] in scripts:
                scripts[row["baseline"]][int(row["seed"])]=row["success"]=="True"
    best=max(scripts,key=lambda name: sum(scripts[name].values()))
    if best!="find_structure" or sum(scripts[best].values())!=7:
        raise ValueError("The preregistered comparison requires the original finding-structure results")
    visual={int(row["seed"]):row["success"]=="True" for row in visual_rows}
    gap=paired_difference(visual,scripts[best])
    wins=groups["visual"]["successes"]
    advantage=wins-groups["shuffle_visual"]["successes"]
    return dict(visual_successes=wins,visual_minus_shuffled=advantage,
                best_script=best,best_script_successes=7,visual_minus_best_script=gap,
                passed=wins>=23 and advantage>=6 and gap["gap"]>=6)


def evaluate_distance(workers, run, seeds, standard, distance, repaired_scene, ground_repair, recovery_harness=None):
    jobs={arm:workers.submit(run_job,run,f"A/{distance}/{arm}",gpu,seeds,standard,distance,
                             repaired_scene=repaired_scene,ground_repair=ground_repair,recovery_harness=recovery_harness)
          for arm,gpu in zip(ARMS,(6,7),strict=True)}
    groups={}
    outcomes={}
    for arm,future in jobs.items():
        rows=future.result()
        outcomes[arm]=rows
        actions,footprint=trace_stats(rows,distance)
        groups[arm]=dict(score(rows),actions=actions)
        if footprint:
            groups[arm]["footprint"]=footprint
    return groups,outcomes


def run_suite(args):
    run=args.run
    protocol_update=args.generalization is not None
    plan,metrics_path=evaluation_inputs(run,protocol_update)
    recovery_harness=run/"recovery-harness" if args.resume else None
    if recovery_harness:
        for manifest in ("b2-recovery.sha256","continuation.sha256"):
            subprocess.run(["sha256sum","--check","--quiet",manifest],cwd=run,check=True)
    repaired_scene="scene_revision" in plan
    if protocol_update and repaired_scene:
        raise ValueError("The main confirmation uses the original v3 scene")
    by_distance={d:[row["seed"] for row in plan["cases"] if row["distance"]==d] for d in DISTANCES}
    weights=args.offline/"room-head.pt"
    offline=args.offline/"offline_gate.json"
    gate=json.loads(offline.read_text())
    if not gate["passed"] or hashlib.sha256(weights.read_bytes()).hexdigest()!=gate["weights_sha256"]:
        raise ValueError("Confirmation room weights do not match the frozen offline record")
    common=["--training-run",str(args.controller),"--room-weights",str(weights),"--batch","3"]
    shared=[*common,"--fixtures",str(run/"frozen/fixtures.json")]
    standard=[*shared,"--oracle-gate",str(args.oracle_gate),"--offline-gate",str(offline),
              "--metrics",str(metrics_path)]
    result={}
    with ThreadPoolExecutor(max_workers=2) as workers:
        for distance in DISTANCES:
            existing=run/f"A-{distance}.json"
            if recovery_harness and existing.exists():
                result[distance]=json.loads(existing.read_text())
                if any(result[distance]["groups"][arm]["n"]!=len(by_distance[distance]) for arm in ARMS):
                    raise ValueError(f"Incomplete saved main result at {distance}")
                continue
            (run/"gpu.progress").write_text(f"A-{distance}\n")
            groups,outcomes=evaluate_distance(workers,run,by_distance[distance],standard,distance,repaired_scene,
                                              protocol_update and distance==60,recovery_harness)
            visual=groups["visual"]["successes"]
            advantage=visual-groups["shuffle_visual"]["successes"]
            result[distance]=dict(groups=groups,advantage=advantage)
            if not protocol_update:
                result[distance]["passed"]=visual>=23 and advantage>=6
            elif distance==30:
                decision=main_decision(run,groups,outcomes["visual"])
                save(run/"main-decision.json",decision)
                result[distance]["passed"]=decision["passed"]
            save(run/f"A-{distance}.json",result[distance])
            verify(run)

        (run/"gpu.progress").write_text("B\n")
        shared_seeds=by_distance[30]

        def ablation_series(gpu, names):
            results={}
            for name in names:
                if recovery_harness and name=="b2":
                    while not (run/"b2-recovery.status").is_file():
                        sleep(30)
                rows=run_job(run,f"B/{name}",gpu,shared_seeds,shared,repaired_scene=repaired_scene,
                             recovery_harness=recovery_harness)
                results[name]=score(rows)
                save_or_verify(run/f"B-{name}.json",results[name])
            return results

        b2=workers.submit(ablation_series,7,("b2",))
        if recovery_harness:
            ablations=workers.submit(ablation_series,6,("b1","b3")).result()
            ablations.update(b2.result())
        else:
            ablations=workers.submit(ablation_series,6,("b1",)).result()
            ablations.update(b2.result())
            ablations.update(ablation_series(6,("b3",)))
        verify(run)
        (run/"gpu.progress").write_text("C\n")
        pairs=plan["twin_seeds"]
        jobs=[workers.submit(run_job,run,f"C/twin-{gpu}",gpu,pairs[index::2],shared,repaired_scene=repaired_scene,
                             recovery_harness=recovery_harness)
              for index,gpu in enumerate((6,7))]
        twins=twin_summary([r for job in jobs for r in job.result()],plan)
        extra=None
        if protocol_update:
            (run/"gpu.progress").write_text("generalization-30\n")
            extra_run=args.generalization
            extra_plan,extra_gate=evaluation_inputs(extra_run,True)
            if extra_plan.get("scene_revision")!="staggered_screens_with_blind_approach":
                raise ValueError("Expected the existing second repaired scene and cohort")
            seeds=[row["seed"] for row in extra_plan["cases"] if row["distance"]==30]
            extra_options=[*common,"--fixtures",str(extra_run/"frozen/fixtures.json"),"--oracle-gate",
                           str(args.oracle_gate),"--offline-gate",str(offline),"--metrics",str(extra_gate)]
            extra,_=evaluate_distance(workers,extra_run,seeds,extra_options,30,True,False,recovery_harness)
            save_or_verify(extra_run/"A-30.json",extra)
            verify(extra_run)
    verify(run)
    save(run/"confirmation-decision.json",dict(frozen_commit=plan["frozen_commit"],
         scene_revision=plan.get("scene_revision","original"),A=result,B=ablations,C=twins,
         main_passed=result[30]["passed"],generalization_30=extra))
    (run/"gpu.progress").write_text("complete\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","controller","offline","oracle-gate"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--generalization",type=Path)
    parser.add_argument("--resume",action="store_true")
    run_suite(parser.parse_args())
