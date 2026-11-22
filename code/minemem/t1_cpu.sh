#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$project/env.sh"
mode="${1:-}"
output="${2:-}"
workers="${3:-32}"
[[ $output == "$WORK/"* && $workers =~ ^[1-9][0-9]?$ && $workers -le 32 ]]
shift 3
entry_args=()
case "$mode" in
    acceptance) entry=t1_v4.py ;;
    protected) entry=t1_respawn.py; entry_args=(acceptance) ;;
    collect) entry=t1_collect.py ;;
    *) echo 'Usage: t1_cpu.sh {acceptance|protected|collect} OUTPUT WORKERS [options]' >&2; exit 2 ;;
esac

canonical="$MINESTUDIO_DIR"
prefix="$WORK/envs/mcvla"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
export JAVA_HOME="$prefix" PATH="$prefix/bin:$PATH" PYTHONPATH="$WORK/code/JarvisVLA"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR' -XX:ActiveProcessorCount=2"
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$WORK/runtime/java-home" "$output"
children=()
for ((i=0; i<workers; i++)); do
    worker="$(printf '%s/worker-%03d' "$output" "$i")"
    mkdir -p "$worker/minestudio"
    ln -s "$canonical/engine" "$worker/minestudio/engine"
    (
        export MINESTUDIO_DIR="$worker/minestudio" MESA_SHADER_CACHE_DIR="$worker"
        cd "$worker"
        exec xvfb-run -a -e "$worker/xvfb.log" "$prefix/bin/python" -B -u \
            "$project/code/minemem/$entry" "${entry_args[@]}" "$worker" --worker "$i" --workers "$workers" "$@"
    ) >"$worker/worker.log" 2>&1 &
    children+=("$!")
    sleep 0.15
done
status=0
for child in "${children[@]}"; do
    wait "$child" || status=1
done
printf '%s\n' "$status" >"$output/exit_status"
exit "$status"
