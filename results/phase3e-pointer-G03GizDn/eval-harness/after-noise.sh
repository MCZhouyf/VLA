#!/usr/bin/env bash
set -euo pipefail
run="$1"
noise="$2"
shift 2
export WORK="$(realpath "$run/../..")" TZ=Asia/Shanghai HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$run/frozen-source/code/minemem:$WORK/code/JarvisVLA"
while ! test -s "$run/cpu.status"; do sleep 20; done
test "$(cat "$run/cpu.status")" = 0
printf 'waiting-for-pose-sweep\n' > "$run/gpu.progress"
while ! test -s "$noise/gpu.status"; do sleep 30; done
(cd "$run" && sha256sum --check --quiet pointer-weights.sha256)
exec "$WORK/envs/mcvla-phase3/bin/python" -B -u "$run/eval-harness/t1_phase3e_pointer_confirm.py" --run "$run" "$@"
