#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf
source "$WORK/env.sh"
run=/share/project/zhoubowen/zhy/zyf/results/phase3-speed-xOn8Q5je
trap 'code=$?; printf "%s\n" "$code" >"$run/rollout_probe.status"' EXIT
out="$run/rollout-probe"
mkdir -p "$out/minestudio"
ln -s "$MINESTUDIO_DIR/engine" "$out/minestudio/engine"
db=$(mktemp -d /dev/shm/minemem-t1-probe-XXXXXXXX)
ln -s "$db" "$out/minestudio/$(hostname)_tmp"
export MINESTUDIO_DIR="$out/minestudio" MESA_SHADER_CACHE_DIR="$out"
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1
export JAVA_HOME="$WORK/envs/mcvla" PATH="$WORK/envs/mcvla/bin:$PATH"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR' -XX:ActiveProcessorCount=2"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$WORK/code/JarvisVLA:$WORK/code/minemem"
cd "$out"
xvfb-run -a -e "$out/xvfb.log" "$WORK/envs/mcvla-phase3/bin/python" -B -u \
  "$run/rollout_probe.py" "$run/screened_fixtures.json"
