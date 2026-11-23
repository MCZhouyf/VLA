#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run=/share/project/zhoubowen/zhy/zyf/results/phase3-speed-xOn8Q5je
forward="$WORK/results/t1-review-A8vp1K1p/forward-paired/worker-000/forward.csv"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/spatial_eval.status"' EXIT
until [[ -s "$run/spatial.status" ]]; do sleep 45; done
[[ $(cat "$run/spatial.status") == 0 ]]

wait_for_episodes() {
    local name="$1" expected="$2" file="$run/$1/episodes.csv"
    while true; do
        if [[ -s "$file" && $(wc -l < "$file") -ge $expected ]]; then
            return
        fi
        if [[ -s "$run/no_memory.status" && $(cat "$run/no_memory.status") != 0 ]]; then
            printf 'No-memory training exited before %s\n' "$name" >&2
            return 1
        fi
        if [[ -s "$run/no_memory_eval.status" ]]; then
            printf 'No-memory evaluation exited before %s completed\n' "$name" >&2
            return 1
        fi
        sleep 45
    done
}

wait_for_episodes eval-no_memory-nav 11
wait_for_episodes eval-no_memory-nav_offset 21
if [[ -s "$run/no_memory_eval.status" && $(cat "$run/no_memory_eval.status") != 0 ]]; then
    printf 'No-memory evaluation stopped. Spatial evaluation was held.\n' >&2
    exit 1
fi
date -Is
bash code/minemem/t1_screened_eval.sh d30 spatial "$run" "$forward"
wait_for_episodes eval-no_memory-d30 31
bash code/minemem/t1_screened_eval.sh remaining spatial "$run" "$forward"
