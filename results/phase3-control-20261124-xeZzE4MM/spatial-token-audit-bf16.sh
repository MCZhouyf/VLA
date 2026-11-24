#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
source env.sh
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'code=$?; printf "%s\n" "$code" >"$run/spatial-token-audit-bf16.status"' EXIT
while [[ ! -s "$run/diagnostic-d30.status" ]]; do
    if ! kill -0 "$(cat "$run/diagnostic-d30.pid")" 2>/dev/null; then exit 1; fi
    sleep 10
done
[[ $(cat "$run/diagnostic-d30.status") == 0 ]]
step="$(tail -n 1 "$run/repaired/train-spatial/loss.csv" | cut -d, -f1)"
[[ $step =~ ^[0-9]+$ && $step -lt 2400 ]]
exec 9>"$TMPDIR/phase1-vllm-gpu-7.lock"
flock -n 9
"$WORK/envs/mcvla/bin/python" "$WORK/code/minemem/watch_phase0.py" --check-idle 7
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader)"
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]]
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$WORK/code/minemem:$WORK/code/JarvisVLA"
timeout --kill-after=5s 31m "$WORK/envs/mcvla-phase3/bin/python" -B -u "$run/spatial-token-audit-bf16.py" "$run" &
audit="$!"
while kill -0 "$audit" 2>/dev/null; do
    step="$(tail -n 1 "$run/repaired/train-spatial/loss.csv" | cut -d, -f1)"
    if [[ $step =~ ^[0-9]+$ && $step -ge 2900 ]]; then
        kill -TERM "$audit" 2>/dev/null || true
        wait "$audit" || true
        echo 'Auxiliary audit released GPU 7 before the gate and formal stages.'
        exit 75
    fi
    sleep 10
done
wait "$audit"
