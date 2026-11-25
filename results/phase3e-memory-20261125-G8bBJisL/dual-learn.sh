#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$run/dual-learn.status"' EXIT
while [[ ! -s "$run/dual-collect.status" ]]; do sleep 20; done
[[ $(cat "$run/dual-collect.status") == 0 ]]
source env.sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$run/dual-vision-code:$WORK/code/minemem"
python="$WORK/envs/mcvla-phase3/bin/python"
(
  exec 9>"$TMPDIR/phase1-vllm-gpu-6.lock"
  flock -w 30 9
  "$WORK/envs/mcvla/bin/python" -B code/minemem/watch_phase0.py --check-idle 6
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)"
  "$python" -B -u "$run/dual-vision-code/t1_dual_probe.py" cache \
    --collection "$run/dual-data" --vision "$WORK/models/JarvisVLA-Qwen2-VL-7B" \
    --output "$run/dual-features" --device cuda
)
export CUDA_VISIBLE_DEVICES=
"$python" -B -u "$run/dual-vision-code/t1_dual_probe.py" train \
  --cache "$run/dual-features" --initial "$run/probe/waypoint-memory.pt" --output "$run/dual-selection"
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-dual-eval-XXXXXXXX)"
export T1_CPU_DISPLAY_BASE=7200 T1_CPU_LAUNCH_INTERVAL=0.8
bash "$run/dual-snapshot/code/minemem/t1_cpu.sh" dual-enclosure "$run/dual-eval" 4 \
  --split test --pairs 4 --predictions "$run/dual-selection/test-selections.json" \
  --fixtures "$run/dual-data/test"
