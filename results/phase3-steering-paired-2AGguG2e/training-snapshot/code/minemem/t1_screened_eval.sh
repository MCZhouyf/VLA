#!/usr/bin/env bash
set -euo pipefail

project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$project/env.sh"
if (( $# != 4 )); then
    echo 'Usage: t1_screened_eval.sh {gates|d30|remaining|all} VARIANT RUN FORWARD_CSV' >&2
    exit 2
fi
stage="$1"
variant="$2"
run="$3"
forward_control="$4"
[[ $run == "$WORK/results/"* ]]
case "$variant" in
    no_memory|long) gpu=6 ;;
    spatial) gpu=7 ;;
    *) exit 2 ;;
esac
[[ -s $run/train-$variant/summary.json ]]
export T1_LOCAL_DATABASE_ROOT="${T1_LOCAL_DATABASE_ROOT:-$(mktemp -d /dev/shm/minemem-t1-eval-XXXXXXXX)}"
runner="$project/code/minemem/t1_exec.sh"
nav="$run/eval-no_memory-nav/episodes.csv"
offset="$run/eval-no_memory-nav_offset/episodes.csv"
snav="$run/eval-spatial-nav/episodes.csv"
soffset="$run/eval-spatial-nav_offset/episodes.csv"
seeds="$WORK/results/phase1_t1_seeds.json"
python="$WORK/envs/mcvla-phase3/bin/python"

if [[ $stage == gates ]]; then
    [[ $variant == no_memory || $variant == spatial ]]
    gpu="${T1_NAVIGATION_GPU:-$gpu}"
    [[ $gpu =~ ^[67]$ ]]
    gate_options=()
    if [[ ${T1_CAPTURE_NAVIGATION_ACTIONS:-0} == 1 ]]; then
        gate_options+=(--control-trace)
    fi
    for mode in nav nav_offset; do
        bash "$runner" eval "$gpu" "$run/eval-$variant-$mode" \
            --training-run "$run/train-$variant" --mode "$mode" "${gate_options[@]}"
    done
    exit
fi

test -s "$run/structure-audit.json"
case "$stage" in
    d30) distances=(30) ;;
    remaining) distances=(10 60 100) ;;
    all) [[ $variant == long ]]; distances=(30 10 60 100) ;;
    *) exit 2 ;;
esac

selected="$("$python" -B - "$seeds" "$run" "$stage" <<'PY'
import csv
import json
import sys
from pathlib import Path

seed_path, run, stage = sys.argv[1:]
run = Path(run)
audit = json.loads((run / "structure-audit.json").read_text())
if any(not audit[str(d)]["passed"] for d in (30, 60, 100)):
    sys.exit("The screened structure audit has not cleared all distant archives")
seeds = json.loads(Path(seed_path).read_text())["formal"]

def reaches(model):
    with (run / f"eval-{model}-d30/episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30 or {int(row["seed"]) for row in rows} != set(seeds):
        sys.exit(f"{model} does not have all 30 fixed seeds")
    return sum(row["success"] == "True" for row in rows)

if stage in ("remaining", "all") and reaches("no_memory") >= 6:
    sys.exit("No-memory reached 20% or more at 30 blocks")
if stage == "all" and reaches("spatial") <= 15:
    sys.exit("Spatial memory did not exceed 50% at 30 blocks")
print(*seeds)
PY
)"
read -ra frozen_seeds <<<"$selected"
[[ "${#frozen_seeds[@]}" == 30 ]]
bash "$runner" eval "$gpu" "$run/eval-$variant-$stage" \
    --training-run "$run/train-$variant" --mode formal --distances "${distances[@]}" \
    --seeds-file "$seeds" --seed "${frozen_seeds[@]}" --rollout-batch-size 3 \
    --screened-fixtures "$run/screened_fixtures.json" --no-memory-gate "$nav" \
    --offset-gate "$offset" --spatial-gate "$snav" --spatial-offset-gate "$soffset" \
    --forward-control "$forward_control"
