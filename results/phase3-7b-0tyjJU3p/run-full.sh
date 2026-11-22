#!/usr/bin/env bash
set -euo pipefail
run="$1"
cd "$WORK"
trap 'printf "%s\n" "$?" >"$run/full.status"' EXIT
until [[ -s $run/after-smoke.status ]]; do sleep 45; done
[[ "$(cat "$run/after-smoke.status")" == 0 ]]
python3 - "$run/navigation-gate/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    gate=json.load(source)
assert gate["passed"] and gate["successes"] >= 8
PY
full="$run/full"
mkdir -p "$full"
frozen="$WORK/results/phase1_t1_seeds.json"
fixtures="$WORK/results/t1-respawn-tTWwXxTb/verified-acceptance"
index="$WORK/data/t1-v4-L6ysa9jV/index-full/dataset.jsonl"
quantized="$(python3 - "$run/smoke-summary.json" <<'PY'
from pathlib import Path
import json, sys
summary=json.loads(Path(sys.argv[1]).read_text())
print(int(any(x['quantized'] for x in summary.values())))
PY
)"
flags=()
if [[ $quantized == 1 ]]; then flags+=(--quantized); fi
train() {
    local gpu="$1" variant="$2" output="$full/train-$2"
    printf '%s training %s on GPU %s\n' "$(date -Is)" "$variant" "$gpu"
    bash code/minemem/t1_exec.sh train "$gpu" --dataset "$index" \
        --output "$output" --variant "$variant" --epochs 1 "${flags[@]}" >"$output.log" 2>&1
}
train 6 no_memory &
a=$!
train 7 long &
b=$!
first=
wait -n -p first "$a" "$b"
if [[ $first == "$a" ]]; then free_gpu=6; remaining="$b"; else free_gpu=7; remaining="$a"; fi
train "$free_gpu" spatial
wait "$remaining"
printf '%s Full training completed\n' "$(date -Is)"
bash code/minemem/t1_exec.sh eval 6 "$full/navigation-gate" \
    --training-run "$full/train-no_memory" --mode nav >"$full/navigation-gate.log" 2>&1
gate="$full/navigation-gate/episodes.csv"
python3 - "$gate" <<'PY'
from pathlib import Path
import csv, json, sys
path=Path(sys.argv[1])
rows=list(csv.DictReader(path.open()))
assert len(rows)==10 and {int(r['seed']) for r in rows}==set(range(2030990000,2030990010))
successes=sum(r['success']=='True' for r in rows)
result=dict(successes=successes,n=len(rows),required_successes=8,passed=successes>=8)
(path.parent/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if not result['passed']:
    raise SystemExit('The full no-memory checkpoint did not pass visible navigation')
PY
mapfile -t cases < <(python3 - "$frozen" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    seeds=json.load(source)["formal"]
assert len(set(seeds))==30
print(*seeds,sep='\n')
PY
)
[[ ${#cases[@]} -eq 30 ]]
evaluate_lane() {
    local gpu="$1" lane="$2" variant seed output
    for variant in no_memory long spatial; do
        for ((idx=lane; idx<30; idx+=2)); do
            seed="${cases[idx]}"
            output="$full/formal-$variant-$seed"
            printf '%s formal %s seed=%s GPU=%s\n' "$(date -Is)" "$variant" "$seed" "$gpu"
            if ! bash code/minemem/t1_exec.sh eval "$gpu" "$output" \
                --mode formal --training-run "$full/train-$variant" --no-memory-gate "$gate" \
                --acceptance "$fixtures" --seeds-file "$frozen" --seed "$seed" >"$output.log" 2>&1; then
                test -d "$output" && mv "$output" "$full/failed-$variant-$seed"
                mv "$output.log" "$full/failed-$variant-$seed.log"
                bash code/minemem/t1_exec.sh eval "$gpu" "$output" \
                    --mode formal --training-run "$full/train-$variant" --no-memory-gate "$gate" \
                    --acceptance "$fixtures" --seeds-file "$frozen" --seed "$seed" >"$output.log" 2>&1
            fi
        done
    done
}
evaluate_lane 6 0 &
lane_a=$!
evaluate_lane 7 1 &
lane_b=$!
status=0
wait "$lane_a" || status=1
wait "$lane_b" || status=1
[[ $status == 0 ]]
export MPLCONFIGDIR="$WORK/.cache/matplotlib" CUDA_VISIBLE_DEVICES=
envs/mcvla-phase1-vllm/bin/python -B code/minemem/t1_comparison.py "$full/report" \
    --runs "$full/train-no_memory" "$full/train-long" "$full/train-spatial" \
    --frozen "$frozen" --episodes "$full"/formal-*/episodes.csv \
    --zero-shot-summary "$run/public/report/pilot_summary.csv"
