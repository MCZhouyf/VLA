#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
out="$(cd -- "$(dirname -- "$0")" && pwd -P)"
cell="$(dirname "$out")"
paired="$(dirname "$cell")"
prior="$WORK/results/phase3-control-20261124-xeZzE4MM/repaired"
original="$WORK/results/phase3-speed-xOn8Q5je"
seeds="$WORK/results/phase1_t1_seeds.json"
fixtures="$original/screened_fixtures.json"
forward="$WORK/results/t1-review-A8vp1K1p/forward-paired/worker-000/forward.csv"
trap 'rc=$?; printf "%s\n" "$rc" >"$out/status"' EXIT
while [[ ! -s "$cell/confirm.status" ]]; do sleep 20; done
[[ $(cat "$cell/confirm.status") == 0 ]]
permission="$(python3 -B - "$cell/confirm-verdict.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if r["spatial_navigation_passed"]:
    print("pass")
else:
    print("held")
PY
)"
if [[ $permission != pass ]]; then
    printf 'held\n' >"$out/activation.txt"
    exit 0
fi
while [[ ! -s "$paired/continuation-spatial.status" ]]; do sleep 20; done
[[ $(cat "$paired/continuation-spatial.status") == 0 ]]
test "$cell/adapter/adapter_model.safetensors" -ef "$prior/train-no_memory/adapter/adapter_model.safetensors"
test "$cell/spatial.safetensors" -ef "$prior/train-spatial/spatial.safetensors"
list="$(python3 -B - "$seeds" "$original/structure-audit.json" "$out/checkpoints.json" "$prior" "$cell" <<'PY'
import json,sys
from pathlib import Path
seed_path,audit_path,output,prior,cell=map(Path,sys.argv[1:])
audit=json.loads(audit_path.read_text())
if any(not audit[str(distance)]["passed"] for distance in (30,60,100)):
    raise SystemExit("Frozen structure audit has not passed")
seeds=json.loads(seed_path.read_text())["formal"]
if len(seeds)!=30 or len(set(seeds))!=30:
    raise SystemExit("Expected all frozen 30-block scenarios")
record=dict(control_lora_in_both=str(prior/"train-no_memory"),
            spatial_memory_module_source=str(prior/"train-spatial"),
            assembled_spatial=str(cell),
            control_lora_shared=True,
            source_navigation_verdict=str(cell/"confirm-verdict.json"))
with output.open("x") as stream:
    json.dump(record,stream,indent=2);stream.write("\n")
print(*seeds)
PY
)"
read -ra frozen <<<"$list"
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-paired-d30-XXXXXXXX)"
printf 'started\n' >"$out/activation.txt"
pids=()
for variant in no_memory spatial; do
  if [[ $variant == no_memory ]]; then
    gpu=6
    weights="$prior/train-no_memory"
  else
    gpu=7
    weights="$cell"
  fi
  bash code/minemem/t1_exec.sh eval "$gpu" "$out/$variant" --mode formal --training-run "$weights" \
    --distances 30 --seed "${frozen[@]}" --seeds-file "$seeds" --rollout-batch-size 3 \
    --screened-fixtures "$fixtures" --no-memory-gate "$prior/eval-no_memory-nav/episodes.csv" \
    --offset-gate "$prior/eval-no_memory-nav_offset/episodes.csv" --forward-control "$forward" \
    --spatial-gate "$cell/nav-all.csv" --spatial-offset-gate "$cell/nav_offset-all.csv" \
    >"$out/$variant.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  printf '%s\n' "$pid" >"$out/$variant.pid"
done
code=0
for pid in "${pids[@]}"; do wait "$pid" || code=1; done
[[ $code == 0 ]]
python3 -B - "$out" "$seeds" <<'PY'
import csv,json,sys
from pathlib import Path
out,seed_path=map(Path,sys.argv[1:])
frozen=set(json.loads(seed_path.read_text())["formal"])
successes={}
for variant in ("no_memory","spatial"):
  with (out/variant/"episodes.csv").open(newline="") as stream:
    rows=list(csv.DictReader(stream))
  if len(rows)!=30 or {int(r["seed"]) for r in rows}!=frozen:
    raise ValueError("Result did not cover each frozen seed exactly once")
  if any(r["mode"]!="formal" or r["model"]!=variant or float(r["d"])!=30 for r in rows):
    raise ValueError("Result is not the planned 30-block formal comparison")
  successes[variant]=sum(r["success"]=="True" for r in rows)
result=dict(formal_d30_successes=successes,episodes_per_variant=30,
            proceed=successes["no_memory"]<6 and successes["spatial"]>15)
with (out/"early-decision.json").open("x") as stream:
    json.dump(result,stream,indent=2);stream.write("\n")
print(json.dumps(result))
PY
date -Is
