#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai MINEMEM_PROXY=
cd "$WORK"
source env.sh
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
out="$(cd -- "$(dirname -- "$0")" && pwd -P)"
export PYTHONPATH="$out/snapshot:$WORK/code/minemem:$WORK/code/JarvisVLA"
trap 'rc=$?; printf "%s
" "$rc" >"$out/regression.status"' EXIT
"$WORK/envs/mcvla-phase3/bin/python" -B -u "$out/snapshot/test_t1_resume.py" \
  --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B"
