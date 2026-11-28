#!/usr/bin/env bash
set -euo pipefail
run="$1"
shift
export WORK="$(realpath "$run/../../..")" TZ=Asia/Shanghai
source "$run/frozen-source/env.sh"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$run/frozen-source/code/minemem:$WORK/code/JarvisVLA"
while ! test -s "$run/gpu.status"; do sleep 15; done
printf 'Original evaluator exited with %s\n' "$(cat "$run/gpu.status")"
(cd "$run" && sha256sum --check --quiet b2-recovery.sha256 && sha256sum --check --quiet continuation.sha256)
exec "$WORK/envs/mcvla-phase3/bin/python" -B -u "$run/recovery-harness/t1_phase3e_confirmation_run.py" --run "$run" --resume "$@"
