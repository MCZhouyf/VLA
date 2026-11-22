#!/usr/bin/env bash
set -euo pipefail
run="$1"
cd "$WORK"
while true; do
    for prior in no_memory spatial; do
        if [[ -s $run/smoke-$prior.status ]]; then
            gpu="$(cat "$run/smoke-$prior.gpu")"
            if envs/mcvla/bin/python code/minemem/watch_phase0.py --check-idle "$gpu" >/dev/null; then
                exec bash "$run/smoke-variant.sh" "$run" "$gpu" long
            fi
        fi
    done
    sleep 30
done
