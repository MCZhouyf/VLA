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
    paired-steering) entry=t1_paired_steering.py; entry_args=(collect) ;;
    pose-replay) entry=t1_pose_replay.py ;;
    rgb-pose-replay) entry=t1_rgb_pose_replay.py ;;
    dual-enclosure) entry=t1_dual_enclosure.py ;;
    phase3e) entry=t1_phase3e.py ;;
    visibility) entry=t1_shortcut_audit.py; entry_args=(audit) ;;
    forward-control) entry=t1_shortcut_audit.py; entry_args=(navigation) ;;
    *) echo 'Usage: t1_cpu.sh {acceptance|protected|collect|paired-steering|pose-replay|rgb-pose-replay|dual-enclosure|phase3e|visibility|forward-control} OUTPUT WORKERS [options]' >&2; exit 2 ;;
esac

canonical="$MINESTUDIO_DIR"
prefix="$WORK/envs/mcvla"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
export JAVA_HOME="$prefix" PATH="$prefix/bin:$PATH" PYTHONPATH="$WORK/code/JarvisVLA"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR' -XX:ActiveProcessorCount=2"
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$WORK/runtime/java-home" "$output"
children=()
shards=()
if [[ -n ${T1_CPU_SHARDS:-} ]]; then
    read -ra shards <<<"$T1_CPU_SHARDS"
    declare -A listed=()
    for i in "${shards[@]}"; do
        [[ $i =~ ^(0|[1-9][0-9]?)$ && $i -lt $workers && ! -v "listed[$i]" ]]
        listed[$i]=1
    done
else
    for ((i=0; i<workers; i++)); do shards+=("$i"); done
fi
display_base="${T1_CPU_DISPLAY_BASE:-}"
if [[ -n $display_base ]]; then [[ $display_base =~ ^[1-9][0-9]{1,3}$ ]]; fi
for i in "${shards[@]}"; do
    worker="$(printf '%s/worker-%03d' "$output" "$i")"
    mkdir -p "$worker/minestudio"
    ln -s "$canonical/engine" "$worker/minestudio/engine"
    if [[ -n ${T1_LOCAL_DATABASE_ROOT:-} ]]; then
        [[ $T1_LOCAL_DATABASE_ROOT == /dev/shm/minemem-t1-* ]]
        database="$(mktemp -d "$T1_LOCAL_DATABASE_ROOT/worker-XXXXXXXX")"
        ln -s "$database" "$worker/minestudio/$(hostname)_tmp"
    fi
    (
        export MINESTUDIO_DIR="$worker/minestudio" MESA_SHADER_CACHE_DIR="$worker"
        cd "$worker"
        xvfb_args=(-a -e "$worker/xvfb.log")
        if [[ -n $display_base ]]; then xvfb_args+=(-n "$((display_base + i))"); fi
        exec xvfb-run "${xvfb_args[@]}" "$prefix/bin/python" -B -u \
            "$project/code/minemem/$entry" "${entry_args[@]}" "$worker" --worker "$i" --workers "$workers" "$@"
    ) >"$worker/worker.log" 2>&1 &
    children+=("$!")
    sleep "${T1_CPU_LAUNCH_INTERVAL:-0.15}"
done
status=0
for child in "${children[@]}"; do
    wait "$child" || status=1
done
printf '%s\n' "$status" >"$output/exit_status"
exit "$status"
