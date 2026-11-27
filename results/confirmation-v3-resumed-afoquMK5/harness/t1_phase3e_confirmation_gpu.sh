#!/usr/bin/env bash
set -euo pipefail

run="$1"
gpu="$2"
output="$3"
shift 3
[[ $gpu =~ ^[67]$ && $output == "$run/"* ]]
source "$run/frozen-source/env.sh"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
export PYTHONPATH="$run/frozen-source/code/minemem:$WORK/code/JarvisVLA"
prefix="$WORK/envs/mcvla"
export JAVA_HOME="$prefix" PATH="$prefix/bin:$PATH"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR' -XX:ActiveProcessorCount=2"
export LP_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 LIBGL_ALWAYS_SOFTWARE=1
mkdir -p "$TMPDIR" "$WORK/runtime/java-home"
exec 9>"$TMPDIR/phase1-vllm-gpu-$gpu.lock"
flock -E 75 -w 30 9
if ! "$prefix/bin/python" "$run/frozen-source/code/minemem/watch_phase0.py" --check-idle "$gpu"; then
    exit 75
fi
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
[[ $CUDA_VISIBLE_DEVICES == GPU-* ]]
export CUDA_VISIBLE_DEVICES
mkdir -p "$output/minestudio"
ln -s "$MINESTUDIO_DIR/engine" "$output/minestudio/engine"
database="$(mktemp -d "$T1_LOCAL_DATABASE_ROOT/eval-XXXXXXXX")"
ln -s "$database" "$output/minestudio/$(hostname)_tmp"
export MINESTUDIO_DIR="$output/minestudio" MESA_SHADER_CACHE_DIR="$output"
cd "$output"
exec xvfb-run -a -e "$output/xvfb.log" "$WORK/envs/mcvla-phase3/bin/python" -B -u \
  "$run/harness/t1_phase3e_confirmation_gpu.py" --output "$output" \
  --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B" "$@"
