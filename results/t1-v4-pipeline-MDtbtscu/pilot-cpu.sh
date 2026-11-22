#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf GPU_IDS=6,7
cd "$WORK"
pipeline="$1"
acceptance="$2"
trap 'printf "%s\n" "$?" > "$pipeline/pilot_cpu_status"' EXIT
until [[ -s $acceptance/exit_status ]]; do sleep 20; done
[[ "$(cat "$acceptance/exit_status")" == 0 ]]
frozen="$WORK/results/phase1_t1_seeds.json"
children=()
for seed in 2026112100 2026112101 2026112102 2026112103 2026112104; do
    for group in 100 60 10,30; do
        IFS=, read -a distances <<< "$group"
        run="$pipeline/cpu-$seed-$group"
        bash code/minemem/t1_exec.sh pilot "$run" --acceptance "$acceptance" \
            --seeds-file "$frozen" --seed "$seed" --models random steve1 vpt \
            --distances "${distances[@]}" >"$run.log" 2>&1 &
        children+=("$!")
        sleep 0.15
    done
done
result=0
for pid in "${children[@]}"; do wait "$pid" || result=1; done
exit "$result"
