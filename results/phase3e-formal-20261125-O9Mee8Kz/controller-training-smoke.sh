#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$run/controller-training-smoke.status"' EXIT
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$run/snapshot/code/minemem:$WORK/code/JarvisVLA"
"$WORK/envs/mcvla-phase3/bin/python" -B -u "$run/controller-training-smoke.py" "$run" \
  "$WORK/models/JarvisVLA-Qwen2-VL-7B"
