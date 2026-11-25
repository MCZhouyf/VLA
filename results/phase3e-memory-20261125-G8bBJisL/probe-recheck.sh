#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
probe="${1:-probe}"
trap 'rc=$?; printf "%s
" "$rc" >"$run/$probe.status"' EXIT
while [[ ! -s "$run/cache.status" ]]; do sleep 15; done
[[ $(cat "$run/cache.status") == 0 ]]
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$run/dual-vision-code:$WORK/code/minemem"
shift || true
"$WORK/envs/mcvla-phase3/bin/python" -B -u "$run/dual-vision-code/t1_memory_probe.py" \
  --cache "$run/cache" --output "$run/$probe" \
  --legacy-weights "$WORK/results/phase3-steering-paired-2AGguG2e/shared-adapter-probe/spatial.safetensors" "$@"
