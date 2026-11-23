#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run=/share/project/zhoubowen/zhy/zyf/results/phase3-speed-xOn8Q5je
old="$WORK/results/phase3-7b-0tyjJU3p/full/train-no_memory"
trap 'code=$?; printf "%s\n" "$code" >"$run/no_memory.status"' EXIT
until [[ -s "$old/summary.json" ]] && ! kill -0 1210874 2>/dev/null; do
    if ! kill -0 1210874 2>/dev/null; then
        echo 'Original training exited without a final summary' >&2
        exit 1
    fi
    sleep 12
done
test -s "$old/adapter/adapter_model.safetensors"
date -Is
bash code/minemem/t1_exec.sh train-resilient 6 "$run/train-no_memory" \
    --dataset "$WORK/data/t1-screened-hzw7nYp7/index-full/dataset.jsonl" \
    --variant no_memory --epochs 1 --archive-cache /dev/shm/minemem-t1-speed-bse1Qtyc/archives \
    --cache-vision --validation-batch-size 4
