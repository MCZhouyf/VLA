#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$project/env.sh"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH="$WORK/code/JarvisVLA"
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1
cd "$WORK"

review="$1"
run="$2"
full="$run/full"
python="$WORK/envs/mcvla-phase3/bin/python"
model="$WORK/models/JarvisVLA-Qwen2-VL-7B"
dataset="$WORK/data/t1-v4-L6ysa9jV/index-full/dataset.jsonl"
seeds="$WORK/results/phase1_t1_seeds.json"
fixtures="$WORK/results/t1-respawn-tTWwXxTb/verified-acceptance"
early="$review/early"
mkdir -p "$early"

wait_checkpoint() {
    local variant="$1" checkpoint="$full/train-$1"
    until [[ -s $checkpoint/summary.json && -s $checkpoint/adapter/adapter_model.safetensors ]]; do
        if [[ -s $run/full.status && $(cat "$run/full.status") != 0 ]]; then
            return 1
        fi
        sleep 45
    done
}

validate_last() {
    local variant="$1" checkpoint="$full/train-$1"
    wait_checkpoint "$variant"
    [[ ! -e $checkpoint/val_loss.csv ]]
    local output="$review/validation-$variant"
    local lane status=0 workers=4
    local children=()
    mkdir -p "$output"
    for ((lane=0; lane<workers; lane++)); do
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$python" -B -u code/minemem/t1_validation.py "$output/shard-$lane.csv" \
            --dataset "$dataset" --checkpoint "$model" --training-run "$checkpoint" \
            --worker "$lane" --workers "$workers" >"$output/shard-$lane.log" 2>&1 &
        children+=("$!")
    done
    for child in "${children[@]}"; do wait "$child" || status=1; done
    [[ $status == 0 ]] || return "$status"
    "$python" -B - "$output" "$checkpoint/val_loss.csv" <<'PY'
import csv, sys
from pathlib import Path

root, output = map(Path,sys.argv[1:])
rows=[]
for file in sorted(root.glob("shard-*.csv")):
    with file.open(newline="") as stream:
        row,=csv.DictReader(stream)
        rows.append(row)
assert len(rows)==4 and all(len({r[key] for r in rows})==1 for key in ("update","device","dtype"))
totals={key:sum(int(r[key]) for r in rows) for key in ("frames","actions","tokens")}
assert totals["frames"]==6500 and totals["actions"]==2268
result=dict(rows[0],**totals)
result["loss"]=sum(float(r["loss"])*int(r["tokens"]) for r in rows)/totals["tokens"]
result["token_accuracy"]=sum(round(float(r["token_accuracy"])*int(r["tokens"])) for r in rows)/totals["tokens"]
result["exact_action_accuracy"]=sum(round(float(r["exact_action_accuracy"])*int(r["actions"])) for r in rows)/totals["actions"]
result["seconds"]=max(float(r["seconds"]) for r in rows)
with output.open("x",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerow(result)
print(result)
PY
}

if [[ ${3:-early} == last-validation ]]; then
    validate_last "$4"
    exit
fi

wait_checkpoint no_memory
checkpoint="$full/train-no_memory"
printf '%s No-memory checkpoint ready\n' "$(date -Is)"
run_navigation() {
    local mode="$1" output="$2" first="$3" count="$4" workers="$5"
    local lane slot shard
    local children=()
    mkdir -p "$output"
    for ((lane=0; lane<workers; lane++)); do
        local cases=()
        for ((slot=lane; slot<count; slot+=workers)); do cases+=("$((first+slot))"); done
        shard="$output/shard-$lane"
        T1_CPU_THREADS=12 bash code/minemem/t1_exec.sh eval-cpu "$shard" \
            --mode "$mode" --training-run "$checkpoint" --seed "${cases[@]}" >"$shard.log" 2>&1 &
        children+=("$!")
    done
    local status=0
    for child in "${children[@]}"; do wait "$child" || status=1; done
    [[ $status == 0 ]] || return "$status"
    "$python" -B - "$output" "$first" "$count" <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
first, count = map(int,sys.argv[2:])
results = []
for file in sorted(path.glob("shard-*/episodes.csv")):
    with file.open(newline="") as stream:
        results.extend(csv.DictReader(stream))
results.sort(key=lambda row:int(row["seed"]))
assert [int(row["seed"]) for row in results] == list(range(first,first+count))
with (path/"episodes.csv").open("x",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
PY
}

run_navigation nav "$early/navigation-original" 2030990000 10 3 >"$early/navigation-original.log" 2>&1 &
original=$!
run_navigation nav_offset "$early/navigation-offset" 2030990100 20 5 >"$early/navigation-offset.log" 2>&1 &
offset=$!
status=0
wait "$original" || status=1
wait "$offset" || status=1
[[ $status == 0 ]]

gate="$early/navigation-original/episodes.csv"
forward="$review/forward-paired/worker-000/forward.csv"
"$python" -B - "$early" "$forward" <<'PY'
import csv, json, sys
from pathlib import Path

root, forward = map(Path, sys.argv[1:])

def counts(path, column, cohort):
    with path.open(newline="") as stream:
        rows=[r for r in csv.DictReader(stream) if r[column]==cohort]
    return sum(r["success"]=="True" for r in rows), len(rows)

old = counts(root/"navigation-original/episodes.csv", "cohort", "aligned")
visible = counts(root/"navigation-offset/episodes.csv", "cohort", "in_view")
hidden = counts(root/"navigation-offset/episodes.csv", "cohort", "out_of_view")
control = counts(forward, "cohort", "in_view")
passed = old[1]==10 and old[0]>=8 and visible[1]==control[1]==15 and hidden[1]==5
passed = passed and visible[0]>=12 and visible[0]-control[0]>=3
result = dict(original=old, in_view=visible, out_of_view=hidden,
              constant_forward_in_view=control, minimum_in_view_margin=3, passed=passed)
(root/"navigation-summary.json").write_text(json.dumps(result, indent=2)+"\n")
print(result, flush=True)
if not passed:
    raise SystemExit("Navigation controls did not clear the formal gate")
PY

audit="$review/visibility_summary.json"
until [[ -s $audit ]]; do sleep 45; done
"$python" -B - "$audit" <<'PY'
import json, sys
from pathlib import Path
audit=json.loads(Path(sys.argv[1]).read_text())
if any(audit[str(d)]["n"]!=30 or audit[str(d)]["scan_rate"]>=.10 for d in (30,60,100)):
    raise SystemExit("The frozen T1 geometry requires revision before formal inference")
PY

mapfile -t cohort < <("$python" -B - "$seeds" <<'PY'
import json, sys
from pathlib import Path
seeds=json.loads(Path(sys.argv[1]).read_text())["formal"]
assert len(seeds)==len(set(seeds))==30
print(*seeds,sep="\n")
PY
)
[[ ${#cohort[@]} == 30 ]]
evaluate_cohort() {
    local prefix="$1"
    shift
    local lane slot out
    local workers=8
    local children=()
    for ((lane=0; lane<workers; lane++)); do
        local cases=()
        for ((slot=lane; slot<30; slot+=workers)); do cases+=("${cohort[slot]}"); done
        out="$early/$prefix-$lane"
        T1_CPU_THREADS=16 bash code/minemem/t1_exec.sh eval-cpu "$out" --mode formal \
            --training-run "$checkpoint" --no-memory-gate "$gate" --acceptance "$fixtures" \
            --seeds-file "$seeds" --seed "${cases[@]}" --distances "$@" >"$out.log" 2>&1 &
        children+=("$!")
    done
    local status=0
    for child in "${children[@]}"; do wait "$child" || status=1; done
    return "$status"
}

evaluate_cohort d30 30
"$python" -B - "$early" "${cohort[@]}" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("d30-*/episodes.csv")):
    with path.open(newline="") as stream:
        rows.extend(csv.DictReader(stream))
assert len(rows)==30 and {r["seed"] for r in rows}==set(sys.argv[2:])
assert all(r["d"]=="30" and r["model"]=="no_memory" for r in rows)
hits=sum(r["success"]=="True" for r in rows)
result=dict(distance=30, successful=hits, n=30, rate=hits/30, proceed=hits<6,
            inference_device="cpu", inference_dtype="float32")
(root/"early_decision.json").write_text(json.dumps(result,indent=2)+"\n")
print(result, flush=True)
if not result["proceed"]:
    raise SystemExit("No-memory check reached the 20 per cent stop threshold")
PY
evaluate_cohort remaining 10 60 100
