#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf GPU_IDS=6,7
cd "$WORK"
pipeline="$1"
data="$2"
status="$pipeline/train_smoke_status"
trap 'printf "%s\n" "$?" >"$status"' EXIT
index="$data/index-smoke/dataset.jsonl"
deadline=$((SECONDS + 28800))
idle() {
    envs/mcvla/bin/python code/minemem/watch_phase0.py --check-idle "$1" >/dev/null
}
until [[ -s $index && -s $data/index-smoke/summary.json ]]; do
    if ! kill -0 "$(cat "$pipeline/collection-sequential.pid")" 2>/dev/null; then
        echo 'Collection exited before a verified smoke dataset was created.'
        exit 1
    fi
    if ((SECONDS >= deadline)); then exit 75; fi
    sleep 45
done
echo "$(date -Is) Verified smoke index is ready"
gpu=
while [[ -z $gpu ]]; do
    count=0
    for id in 6 7; do
        if idle "$id"; then
            count=$((count + 1))
            candidate="$id"
        fi
    done
    if ((count == 1)) || { ((count == 2)) && \
          ! kill -0 "$(cat "$pipeline/pilot-gpu.pid")" 2>/dev/null; }; then
        gpu="$candidate"
        break
    fi
    if ((SECONDS >= deadline)); then
        echo 'No GPU assigned for smoke training before waiting deadline.'
        exit 75
    fi
    sleep 60
done
echo "$(date -Is) Starting 7B LoRA smoke on GPU $gpu"
run="$pipeline/phase3-smoke-bf16"
if ! bash code/minemem/t1_exec.sh train "$gpu" \
     --dataset "$index" --output "$run" --variant no_memory --budget-seconds 3600 \
     >"$pipeline/phase3-smoke-bf16.log" 2>&1; then
    if ! grep -Eqi 'cuda out of memory|OutOfMemoryError' "$pipeline/phase3-smoke-bf16.log"; then
        tail -n 20 "$pipeline/phase3-smoke-bf16.log"
        exit 1
    fi
    echo "$(date -Is) BF16 exceeded memory, retrying with NF4"
    run="$pipeline/phase3-smoke-nf4"
    bash code/minemem/t1_exec.sh train "$gpu" \
        --dataset "$index" --output "$run" --variant no_memory --budget-seconds 3600 --quantized \
        >"$pipeline/phase3-smoke-nf4.log" 2>&1
fi
bash code/minemem/t1_exec.sh eval "$gpu" "$pipeline/phase3-smoke-closed-loop" \
    --training-run "$run" --mode smoke >"$pipeline/phase3-smoke-closed-loop.log" 2>&1
echo "$(date -Is) Training and closed-loop smoke processes completed"
