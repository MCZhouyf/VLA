#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
result="$1"
trap 'printf "%s\n" "$?" > "$result/exit_status"' EXIT
v4="$WORK/results/t1-v4-guarded-K6ARCMSV"
frozen="$WORK/results/phase1_t1_seeds.json"
export MPLCONFIGDIR="$WORK/.cache/matplotlib" CUDA_VISIBLE_DEVICES=

bash code/minemem/t1_cpu.sh protected "$result/acceptance" 32 --fixtures "$v4"
envs/mcvla/bin/python -B code/minemem/t1_respawn_report.py acceptance \
    "$result/acceptance_summary.csv" "$result/acceptance"

children=()
for seed in 2026112100 2026112101 2026112102 2026112103 2026112104; do
    for group in 100 60 10,30; do
        IFS=, read -a distances <<< "$group"
        run="$result/pilot-$seed-$group"
        bash code/minemem/t1_exec.sh protected-pilot "$run" --fixtures "$result/acceptance" \
            --seeds-file "$frozen" --seed "$seed" --models random steve1 vpt \
            --distances "${distances[@]}" >"$run.log" 2>&1 &
        children+=("$!")
        sleep 0.15
    done
done
status=0
for child in "${children[@]}"; do wait "$child" || status=1; done
test "$status" = 0
envs/mcvla-phase1-vllm/bin/python -B code/minemem/t1_respawn_report.py pilot \
    "$result/pilot-report" --seeds-file "$frozen" --inputs "$result"/pilot-*/cliff_T1.csv
