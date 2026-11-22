#!/usr/bin/env bash
set -euo pipefail
run="$1"
cd "$WORK"
while [[ ! -s $run/smoke-no_memory.status && ! -s $run/smoke-long.status ]]; do sleep 30; done
for gpu in 6 7; do
    if envs/mcvla/bin/python code/minemem/watch_phase0.py --check-idle "$gpu" >/dev/null; then
        exec bash "$run/smoke-variant.sh" "$run" "$gpu" spatial
    fi
done
for n in $(seq 1 20); do
    sleep 15
    for gpu in 6 7; do
        if envs/mcvla/bin/python code/minemem/watch_phase0.py --check-idle "$gpu" >/dev/null; then
            exec bash "$run/smoke-variant.sh" "$run" "$gpu" spatial
        fi
    done
done
printf '75\n' >"$run/smoke-spatial.status"
