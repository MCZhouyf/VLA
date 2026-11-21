#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$project/env.sh"

workers="${1:-32}"
seeds="${2:-50}"
if [[ ! $workers =~ ^[1-9][0-9]?$ || ! $seeds =~ ^[1-9][0-9]*$ ]] ||
   (( workers > 32 || workers > seeds )); then
    echo 'Usage: phase1_teacher.sh [WORKERS <= 32] [SEEDS PER LEVEL]' >&2
    exit 2
fi
canonical="$MINESTUDIO_DIR"
prefix="$WORK/envs/mcvla"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
export JAVA_HOME="$prefix" PATH="$prefix/bin:$PATH" PYTHONPATH="$WORK/code/JarvisVLA"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR' -XX:ActiveProcessorCount=2"
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$TMPDIR" "$WORK/runtime/java-home" "$WORK/results"
result="$(mktemp -d "$WORK/results/phase1-teacher-XXXXXXXX")"
printf '%s\n' "$result"
declare -a children=()
for ((i = 0; i < workers; i++)); do
    output="$(printf '%s/worker-%03d' "$result" "$i")"
    mkdir -p "$output/minestudio"
    ln -s "$canonical/engine" "$output/minestudio/engine"
    (
        export MINESTUDIO_DIR="$output/minestudio" MESA_SHADER_CACHE_DIR="$output"
        cd "$output"
        exec xvfb-run -a -e "$output/xvfb.log" "$prefix/bin/python" -u \
            "$project/code/minemem/phase1_t1.py" "$output" \
            --worker "$i" --workers "$workers" --seeds "$seeds"
    ) >"$output/worker.log" 2>&1 &
    children+=("$!")
    sleep 0.15
done

status=0
for child in "${children[@]}"; do
    wait "$child" || status=1
done
exit "$status"
