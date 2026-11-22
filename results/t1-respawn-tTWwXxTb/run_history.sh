#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
result="$1"
trap 'printf "%s\n" "$?" >"$result/history_status"' EXIT
v4="$WORK/results/t1-v4-guarded-K6ARCMSV"
frozen="$WORK/results/phase1_t1_seeds.json"
export MPLCONFIGDIR="$WORK/.cache/matplotlib" CUDA_VISIBLE_DEVICES=
children=()
for pair in 10,2026112100 100,2026112104; do
    IFS=, read -r d seed <<< "$pair"
    run="$result/history-d$d-seed$seed"
    bash code/minemem/t1_exec.sh historical-pilot "$run" --fixtures "$v4" \
        --seeds-file "$frozen" --seed "$seed" --models random steve1 vpt --distances "$d" \
        >"$run.log" 2>&1 &
    children+=("$!")
done
for seed in 2026112140 2026112143; do
    run="$result/legacy-death-$seed"
    bash code/minemem/t1_exec.sh death-replay "$run" --fixtures "$v4" \
        --seed "$seed" --distances 100 >"$run.log" 2>&1 &
    children+=("$!")
done
status=0
for child in "${children[@]}"; do wait "$child" || status=1; done
test "$status" = 0
for pair in 10,2026112100 100,2026112104; do
    IFS=, read -r d seed <<< "$pair"
    key="d$d-seed$seed"
    run="$result/history-$key"
    envs/mcvla-phase1-vllm/bin/python -B code/minemem/t1_respawn_report.py trajectory \
      "$run/vpt-trajectory.svg" --config "$run/$key.json" --outbound "$run/$key-outbound.csv" \
      --trace "$run/$key-vpt-return.csv.gz"
done
