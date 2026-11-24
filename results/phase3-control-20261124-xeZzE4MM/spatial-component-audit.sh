#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
source env.sh
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'code=$?; printf "%s\n" "$code" >"$run/spatial-component-audit.status"' EXIT
[[ $(cat "$run/diagnostic-d30.status") == 0 && $(cat "$run/spatial-token-audit-bf16.status") == 0 ]]
step="$(tail -n 1 "$run/repaired/train-spatial/loss.csv" | cut -d, -f1)"
[[ $step =~ ^[0-9]+$ && $step -lt 2900 && $(cat "$run/spatial-greedy-audit.status") == 0 ]]
exec 9>"$TMPDIR/phase1-vllm-gpu-7.lock"
flock -n 9
"$WORK/envs/mcvla/bin/python" "$WORK/code/minemem/watch_phase0.py" --check-idle 7
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader)"
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]]
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$WORK/code/minemem:$WORK/code/JarvisVLA"
timeout --kill-after=5s 3m "$WORK/envs/mcvla-phase3/bin/python" -B -u "$run/spatial-component-audit.py" "$run"
