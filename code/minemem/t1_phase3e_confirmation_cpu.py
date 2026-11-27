"""Plan and qualify unseen worlds using the frozen T1 v3 teacher and five checks."""

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

from phase1_metrics import wilson
from t1_phase3e import (BASELINES, DISTANCES, FIELDS, PROTOCOL, FirstArrivalArena,
                        acceptance_case, acceptance_outbound, configure_rendering, frozen_gold_slots)


FROZEN = "ed550b0ee9d1ef1226e762d00b35d9e7cc5d1fee"
SEED_NAMESPACE = f"MineMem confirmation {FROZEN} 2026-11-27"


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def existing_seeds(checkout):
    paths = subprocess.check_output(["git", "-C", str(checkout), "ls-files", "-z", "results"]).decode().split("\0")[:-1]
    worlds, scenarios = set(), set()
    sources = []

    def remember(value, dest):
        if isinstance(value, int) and not isinstance(value, bool):
            dest.add(value)
        elif isinstance(value, str) and re.fullmatch(r"-?\d+", value):
            dest.add(int(value))
        elif isinstance(value, list):
            for item in value:
                remember(item, dest)

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("world", "world_seed", "world_seeds"):
                    remember(item, worlds)
                if key in ("seed", "scene_seed", "scenario_seed", "scenario_seeds", "formal"):
                    remember(item, scenarios)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for name in paths:
        path = checkout / name
        suffix = path.suffix
        structured_lines = ("index" in name or "metadata" in name) and name.endswith((".jsonl", ".jsonl.gz"))
        score_table = suffix == ".csv" and not re.search(r"(?:d\d+-seed\d+|world\d+)", path.name)
        if suffix != ".json" and not structured_lines and not score_table:
            continue
        raw = path.read_bytes()
        if not re.search(rb'(?:"|^|,)(?:world|world_seed|world_seeds|scene_seed|scenario_seed|formal|seed)(?:"|,|\r?\n)',
                         gzip.decompress(raw) if name.endswith(".gz") else raw, re.M):
            continue
        if suffix == ".json":
            visit(json.loads(raw))
        elif structured_lines:
            for line in (gzip.decompress(raw) if name.endswith(".gz") else raw).splitlines():
                if line:
                    visit(json.loads(line))
        else:
            for row in csv.DictReader(raw.decode().splitlines()):
                visit(row)
        sources.append(dict(path=name, sha256=hashlib.sha256(raw).hexdigest()))
    return worlds, scenarios, sources


def plan(checkout, output, repaired=False):
    git=["git","-C",str(checkout)]
    inventory=subprocess.check_output([*git,"rev-parse","HEAD"]).decode().strip()
    if repaired:
        from t1_phase3e_repaired_scene import SCENE_REVISION

        subprocess.run([*git,"merge-base","--is-ancestor",FROZEN,inventory],check=True)
        revision=SCENE_REVISION
    else:
        if inventory!=FROZEN:
            raise ValueError("Generate the original plan against the frozen archive")
        revision="original"
    worlds, scenes, sources = existing_seeds(checkout)
    namespace=SEED_NAMESPACE if not repaired else f"{SEED_NAMESPACE} {revision} {inventory}"
    rng = random.Random(namespace)
    occupied_worlds, occupied_scenes = worlds.copy(), scenes.copy()
    cases, balance_attempts = [], {}
    for distance in DISTANCES:
        while True:
            seeds = rng.sample(range(1_800_000_000,1_801_000_000),30)
            balance_attempts[distance] = balance_attempts.get(distance,0)+1
            if occupied_scenes.intersection(seeds):
                continue
            try:
                slots = frozen_gold_slots(seeds)
            except ValueError:
                continue
            break
        occupied_scenes.update(seeds)
        for ordinal, (seed, slot) in enumerate(zip(seeds,slots,strict=True)):
            world = rng.randrange(88_000_000,89_000_000)
            while world in occupied_worlds:
                world = rng.randrange(88_000_000,89_000_000)
            occupied_worlds.add(world)
            cases.append(dict(distance=distance, seed=seed, world_seed=world, goal_slot=slot, ordinal=ordinal))
    twins = [row["seed"] for slot in (0,1) for row in
             [r for r in cases if r["distance"]==30 and r["goal_slot"]==slot][:5]]
    output.mkdir()
    planned=dict(frozen_commit=FROZEN, namespace=namespace, cases=cases,
                 twin_seeds=twins,balance_attempts=balance_attempts)
    if repaired:
        planned.update(scene_revision=revision, inventory_commit=inventory)
        previously_failed=checkout/"results/confirmation-ed550b0-iZY2YtxL/seed-plan.json"
        earlier=json.loads(previously_failed.read_text())["cases"]
        if not (set(row["seed"] for row in earlier)<=scenes and set(row["world_seed"] for row in earlier)<=worlds):
            raise ValueError("The repaired confirmation inventory must contain the entire rejected cohort")
    save(output/"seed-plan.json",planned)
    save(output/"seed-audit.json",dict(frozen_commit=FROZEN, old_world_count=len(worlds),
                                      old_scenario_count=len(scenes), new_world_count=len(cases),
                                      new_scenario_count=len(cases), world_overlap=0, scenario_overlap=0,
                                      old_world_seeds=sorted(worlds), old_scenario_seeds=sorted(scenes), sources=sources))


def worker(output, seed_plan, rank, workers, retry, scene):
    cases = json.loads(seed_plan.read_text())["cases"]
    configure_rendering(output)
    original = output if retry is None else retry
    seen = set()
    if retry is not None and (original/"acceptance.csv").exists():
        with (original/"acceptance.csv").open(newline="") as stream:
            seen = {int(row["seed"]) for row in csv.DictReader(stream) if row["baseline"]=="start_pose" and not row["error"]}
    arena_type=FirstArrivalArena
    if scene != "original":
        from t1_phase3e_repaired_scene import GroundProbe, GroundRepairedArena, RepairedScene

        arena_type={"repaired":RepairedScene,"ground-probe":GroundProbe,
                    "ground-repaired":GroundRepairedArena}[scene]
    with (output/"acceptance.csv").open("x",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(FIELDS)
        for item in cases[rank::workers]:
            distance,seed,world,slot=(item[key] for key in ("distance","seed","world_seed","goal_slot"))
            if seed in seen:
                continue
            arena=None
            try:
                arena=arena_type(world,effects=True)
                layout,poses=acceptance_outbound(arena,seed,distance,slot,output)
                for row in acceptance_case(arena,layout,poses,output):
                    writer.writerow(row)
                    stream.flush()
            except Exception as error:
                print(f"d={distance} seed={seed} {type(error).__name__}: {error}",flush=True)
                writer.writerow((distance,seed,world,slot,*([""]*10),str(error)))
                stream.flush()
            finally:
                if arena is not None:
                    arena.sim.close()


def aggregate(seed_plan, inputs, output):
    output.mkdir()
    cases = json.loads(seed_plan.read_text())["cases"]
    expected = {(row["distance"],row["seed"]):row for row in cases}
    complete, configs = {}, {}
    errors = []
    for source in inputs:
        for path in sorted(source.glob("worker-*/acceptance.csv")):
            partial = {}
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    key = int(row["distance"]),int(row["seed"])
                    planned=expected[key]
                    if int(row["world_seed"])!=planned["world_seed"] or int(row["goal_slot"])!=planned["goal_slot"]:
                        raise ValueError(f"Wrong case assignment in {path}: {key}")
                    if row["error"]:
                        errors.append(dict(source=str(path), distance=key[0], seed=key[1], error=row["error"]))
                    else:
                        bucket=partial.setdefault(key,{})
                        if row["baseline"] in bucket or row["baseline"] not in BASELINES:
                            raise ValueError(f"Invalid baseline coverage for {key} in {path}")
                        bucket[row["baseline"]]=row
            for key,rows in partial.items():
                if set(rows)==set(BASELINES):
                    if key in complete:
                        raise ValueError(f"Repeated complete CPU case {key}")
                    complete[key]=rows
                    configs[f"{key[0]}/{key[1]}"]=json.loads((path.parent/f"d{key[0]}-seed{key[1]}.json").read_text())
    missing=sorted(set(expected)-complete.keys())
    metrics = {}
    for distance in DISTANCES:
        sub=[rows for key,rows in complete.items() if key[0]==distance]
        metrics[distance]={}
        for baseline in BASELINES:
            values=[rows[baseline] for rows in sub]
            wins=sum(row["success"]=="True" for row in values)
            bounds=wilson(wins,len(values)) if values else (0,1)
            metrics[distance][baseline]=dict(n=len(values),successes=wins,rate=wins/len(values) if values else 0,
                 wilson95=bounds,deaths=sum(int(row["deaths"]) for row in values),
                 pose_jumps=sum(int(row["pose_jumps"]) for row in values))
        low, high = metrics[distance]["reverse_route"]["wilson95"]
        metrics[distance]["reverse_route"]["compatible_with_half"] = low<=.50<=high
    passed=(not missing and all(metrics[d]["teacher"]["rate"]>=.95 for d in DISTANCES)
            and all(metrics[d][name]["rate"]<.10 for d in (30,60,100)
                    for name in ("turn_180","scan","find_structure","start_pose"))
            and all(metrics[d]["reverse_route"]["compatible_with_half"] for d in (30,60,100)))
    save(output/"metrics.json",dict(passed=passed,distances=metrics,missing=missing,errors=errors))
    save(output/"fixtures.json",dict(protocol=PROTOCOL,formal=[r["seed"] for r in cases],fixtures=configs))
    with (output/"acceptance.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader()
        for key in expected:
            writer.writerows(complete.get(key,{}).values())
    names=("metrics.json","fixtures.json","acceptance.csv")
    (output/"SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((output/name).read_bytes()).hexdigest()}  {name}\n" for name in names))
    print(f"CPU acceptance complete={len(complete)}/{len(expected)} passed={passed}",flush=True)
    return passed


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    stages=parser.add_subparsers(dest="stage",required=True)
    planning=stages.add_parser("plan")
    planning.add_argument("checkout",type=Path)
    planning.add_argument("output",type=Path)
    planning.add_argument("--repaired-scene",action="store_true")
    cpu=stages.add_parser("worker")
    cpu.add_argument("output",type=Path)
    cpu.add_argument("--plan",type=Path,required=True)
    cpu.add_argument("--worker",type=int,required=True)
    cpu.add_argument("--workers",type=int,required=True)
    cpu.add_argument("--retry-from",type=Path)
    cpu.add_argument("--scene",choices=("original","repaired","ground-probe","ground-repaired"),default="original")
    freeze=stages.add_parser("freeze")
    freeze.add_argument("output",type=Path)
    freeze.add_argument("--plan",type=Path,required=True)
    freeze.add_argument("--inputs",type=Path,nargs="+",required=True)
    args=parser.parse_args()
    if args.stage=="plan":
        plan(args.checkout,args.output,args.repaired_scene)
    elif args.stage=="worker":
        original=args.retry_from/args.output.name if args.retry_from else None
        worker(args.output,args.plan,args.worker,args.workers,original,args.scene)
    else:
        raise SystemExit(0 if aggregate(args.plan,args.inputs,args.output) else 1)
