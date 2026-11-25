#!/usr/bin/env bash
set -euo pipefail
run="$1"
export WORK="${run%/results/*}" TZ=Asia/Shanghai
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-phase3e-glfw-data-XXXXXXXX)"
original="$run/data/train/worker-005"
if [[ ! -f "$original/episodes.jsonl" ]]; then
    retry="$run/train-glfw-retry"
    if T1_CPU_SHARDS=5 T1_CPU_DISPLAY_BASE=7790 bash "$run/snapshot/code/minemem/t1_cpu.sh" phase3e \
       "$retry" 24 --split train >"$run/train-glfw-retry.log" 2>&1; then
        [[ -s "$retry/worker-005/episodes.jsonl" ]]
        mv "$original" "$run/train-worker-005-glfw-crash"
        ln -s "$retry/worker-005" "$original"
    fi
fi
while [[ ! -s "$run/stage-a.status" ]]; do sleep 15; done
[[ $(cat "$run/stage-a.status") != 0 && $(cat "$run/stage-a.progress") == collection ]]
mv "$run/stage-a.status" "$run/collection-first.status"
printf '%s\n' "$$" >"$run/stage-a.pid"
bash "$run/snapshot/code/minemem/t1_phase3e_stage_a.sh" "$run" --resume-collection
