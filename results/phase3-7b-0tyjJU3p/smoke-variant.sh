#!/usr/bin/env bash
set -euo pipefail
run="$1"
gpu="$2"
variant="$3"
cd "$WORK"
prefix="$run/smoke-$variant"
trap 'printf "%s\n" "$?" >"$prefix.status"' EXIT
printf '%s\n' "$gpu" >"$prefix.gpu"
index="$WORK/data/t1-v4-L6ysa9jV/index-smoke/dataset.jsonl"
train() {
    local output="$1"
    shift
    bash code/minemem/t1_exec.sh train "$gpu" --dataset "$index" \
        --output "$output" --variant "$variant" --budget-seconds 3600 "$@" \
        >"$output.log" 2>&1
}
model="$prefix-bf16"
if ! train "$model"; then
    if ! grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$model.log"; then
        tail -n 30 "$model.log"
        exit 1
    fi
    printf '%s BF16 exceeded memory on GPU %s, retrying with NF4\n' "$(date -Is)" "$gpu"
    model="$prefix-nf4"
    train "$model" --quantized
fi
printf '%s\n' "$model" >"$prefix.training-run"
printf 'Training finished at %s, starting MineStudio evaluation\n' "$(date -Is)"
bash code/minemem/t1_exec.sh eval "$gpu" "$prefix-closed-loop" \
    --training-run "$model" --mode smoke >"$prefix-closed-loop.log" 2>&1
