#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
prior="$WORK/results/phase3-control-20261124-xeZzE4MM/repaired"
forward="$WORK/results/t1-review-A8vp1K1p/forward-paired/worker-000/forward.csv"
trap 'rc=$?; printf "%s\n" "$rc" >"$dir/confirm.status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-shared-confirm-XXXXXXXX)"
test "$dir/adapter/adapter_model.safetensors" -ef "$prior/train-no_memory/adapter/adapter_model.safetensors"
test "$dir/spatial.safetensors" -ef "$prior/train-spatial/spatial.safetensors"
sha256sum "$dir/adapter/adapter_model.safetensors" "$dir/spatial.safetensors" >"$dir/checkpoint-sha256.txt"
aligned=(203099000{2..9})
offsets=(20309901{04..19})
bash code/minemem/t1_exec.sh eval 6 "$dir/nav-remainder" --mode nav --training-run "$dir" \
    --seed "${aligned[@]}" --control-trace
bash code/minemem/t1_exec.sh eval 6 "$dir/nav_offset-remainder" --mode nav_offset --training-run "$dir" \
    --seed "${offsets[@]}" --control-trace
export PYTHONPATH="$WORK/code/minemem:$WORK/code/JarvisVLA" CUDA_VISIBLE_DEVICES=
"$WORK/envs/mcvla-phase3/bin/python" -B - "$dir" "$prior" "$forward" <<'PY'
import csv
import json
import sys
from pathlib import Path
from t1_trained_eval import navigation_cases, validate_aligned_gate, validate_offset_gate

directory, prior, forward = map(Path,sys.argv[1:])
merged = {}
for mode in ("nav", "nav_offset"):
    expected={str(seed) for seed, *_ in navigation_cases(mode=="nav_offset")}
    merged[mode]=[]
    fields = None
    for sub in (mode,mode+"-remainder"):
        with (directory/sub/"episodes.csv").open(newline="") as stream:
            reader=csv.DictReader(stream)
            if fields is None:
                fields=reader.fieldnames
            if fields!=reader.fieldnames:
                raise ValueError("Gate CSV headers differ")
            merged[mode].extend(reader)
    rows=merged[mode]
    if len(rows) != len(expected) or {row["seed"] for row in rows}!=expected:
        raise ValueError("Gate sources do not cover unique frozen scenarios")
    if any(row["training_run"]!=str(directory.resolve()) or row["mode"]!=mode for row in rows):
        raise ValueError("Gate sources did not use the same assembled check point")
    with (directory/f"{mode}-all.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fields)
        writer.writeheader()
        writer.writerows(sorted(rows,key=lambda row:row["seed"]))
base=validate_aligned_gate(prior/"eval-no_memory-nav/episodes.csv","no_memory")
if Path(base).resolve()!=(prior/"train-no_memory").resolve():
    raise ValueError("Reference data points to a different control checkpoint")
validate_offset_gate(prior/"eval-no_memory-nav_offset/episodes.csv",base,forward,"no_memory")
errors=[]
try:
    checkpoint=validate_aligned_gate(directory/"nav-all.csv","spatial")
    if Path(checkpoint.resolve() if isinstance(checkpoint,Path) else checkpoint).resolve()!=directory.resolve():
        raise ValueError("Shared LoRA results point to the wrong checkpoint")
except (ValueError,SystemExit) as error:
    errors.append(str(error))
try:
    validate_offset_gate(directory/"nav_offset-all.csv",str(directory.resolve()),forward,"spatial")
except (ValueError,SystemExit) as error:
    errors.append(str(error))
cohorts={}
for rows in merged.values():
    for row in rows:
        count=cohorts.setdefault(row["cohort"],dict(episodes=0,successes=0))
        count["episodes"]+=1
        count["successes"]+=row["success"]=="True"
result=dict(checkpoint=str(directory),verified_no_memory_reference=base,
    cohorts=cohorts,spatial_navigation_passed=not errors,blockers=errors)
with (directory/"confirm-verdict.json").open("x") as stream:
    json.dump(result,stream,indent=2);stream.write("\n")
print(json.dumps(result))
PY
date -Is
