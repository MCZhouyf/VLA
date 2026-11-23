#!/usr/bin/env bash
set -euo pipefail

output="$1"
shift
if "$@"; then
    exit 0
else
    status=$?
fi
if ! compgen -G "$output/checkpoints/step-*/state.pt" >/dev/null; then
    echo "Training exited with status $status before writing a checkpoint" >&2
    exit "$status"
fi
echo "Training exited with status $status; attempting one checkpoint recovery" >&2
exec "$@" --resume
