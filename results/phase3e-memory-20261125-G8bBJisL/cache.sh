#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$run/cache.status"' EXIT
source env.sh
exec 9>"$TMPDIR/phase1-vllm-gpu-6.lock"
flock -w 30 9
"$WORK/envs/mcvla/bin/python" -B code/minemem/watch_phase0.py --check-idle 6
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$run/code:$WORK/code/minemem"
"$WORK/envs/mcvla-phase3/bin/python" -B -u "$run/code/t1_memory_cache.py" \
  --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B" \
  --dataset "$WORK/data/t1-screened-hzw7nYp7" \
  --archive-cache "$(cat "$WORK/results/phase3-speed-xOn8Q5je/disk_cache_path")" \
  --device cuda --output "$run/cache"
