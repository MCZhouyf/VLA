#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
source env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WORK/code/minemem:$WORK/code/JarvisVLA"
out="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$out/status"' EXIT
"$WORK/envs/mcvla/bin/python" -B -u "$out/t1_rgb_anchor_audit.py" \
  --dataset "$WORK/data/t1-screened-hzw7nYp7/index-full/dataset.jsonl" \
  --archive-cache "$(cat "$WORK/results/phase3-speed-xOn8Q5je/disk_cache_path")" \
  --output "$out/heldout"
