#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run=/share/project/zhoubowen/zhy/zyf/results/phase3-speed-xOn8Q5je
trap 'rc=$?; printf "%s\n" "$rc" >"$run/spatial.status"' EXIT
date -Is
bash code/minemem/t1_exec.sh train-resilient 7 "$run/train-spatial" \
    --dataset "$WORK/data/t1-screened-hzw7nYp7/index-full/dataset.jsonl" \
    --archive-cache "$(cat "$run/disk_cache_path")" --variant spatial --epochs 1 \
    --accumulation 4 --seed 20261122 --learning-rate 0.0001 \
    --cache-vision --validation-batch-size 4
