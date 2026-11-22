#!/usr/bin/env bash
set -euo pipefail
run="$1"
cd "$WORK"
trap 'printf "%s\n" "$?" >"$run/smoke-no_memory.status"' EXIT
model="$(cat "$run/smoke-no_memory.training-run")"
while true; do
    for finished in long spatial; do
        if [[ -s $run/smoke-$finished.status ]]; then
            gpu="$(cat "$run/smoke-$finished.gpu")"
            if envs/mcvla/bin/python code/minemem/watch_phase0.py --check-idle "$gpu" >/dev/null; then
                printf '%s\n' "$gpu" >"$run/smoke-no_memory.eval-gpu"
                bash code/minemem/t1_exec.sh eval "$gpu" "$run/smoke-no_memory-closed-loop" \
                    --training-run "$model" --mode smoke >"$run/smoke-no_memory-closed-loop.log" 2>&1
                exit 0
            fi
        fi
    done
    sleep 25
done
