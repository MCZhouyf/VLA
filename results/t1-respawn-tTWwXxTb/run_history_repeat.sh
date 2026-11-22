#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
result="$1"
d="$2"
seed="$3"
key="d$d-seed$seed"
shift 3
v4="$WORK/results/t1-v4-guarded-K6ARCMSV"
frozen="$WORK/results/phase1_t1_seeds.json"
export MPLCONFIGDIR="$WORK/.cache/matplotlib" CUDA_VISIBLE_DEVICES=
children=()
for attempt in "$@"; do
    run="$result/history-$key-repeat-$attempt"
    (
        bash code/minemem/t1_exec.sh historical-pilot "$run" --fixtures "$v4" \
          --seeds-file "$frozen" --seed "$seed" --distances "$d" --models random steve1 vpt
        envs/mcvla-phase1-vllm/bin/python -B code/minemem/t1_respawn_report.py trajectory \
          "$run/vpt-trajectory.svg" --config "$run/$key.json" --outbound "$run/$key-outbound.csv" \
          --trace "$run/$key-vpt-return.csv.gz"
    ) >"$run.log" 2>&1 &
    children+=("$!")
    sleep 8
done
status=0
for child in "${children[@]}"; do wait "$child" || status=1; done
printf '%s\n' "$status" >"$result/history-repeat-$d-status"
exit "$status"
