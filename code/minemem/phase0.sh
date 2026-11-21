#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"

case "${1:-}" in
    check|random|mcu|watch|capacity) mode="$1" ;;
    *) printf '%s\n' 'Usage: phase0.sh {check|random|mcu|watch|capacity [levels...]}' >&2; exit 2 ;;
esac

if [[ $mode == check ]]; then
    printf 'CPUs available: %s\n' "$(nproc)"
    df -h -- "${WORK:-$project}"
    command -v conda Xvfb xvfb-run xauth || true
    if [[ -z ${WORK:-} ]]; then
        printf '%s\n' 'WORK is unset.' >&2
    else
        source "$project/env.sh"
    fi
    gpu_pair="${GPU_IDS:-6,7}"
    if [[ ! $gpu_pair =~ ^[0-9]+,[0-9]+$ || ${gpu_pair%,*} == "${gpu_pair#*,}" ]]; then
        printf '%s\n' 'GPU_IDS must contain two distinct allocated GPU indices.' >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="$gpu_pair"
    nvidia-smi -i "$gpu_pair" --query-gpu=index,name,memory.total --format=csv
    exit
fi

source "$project/env.sh"
prefix="$WORK/envs/mcvla"
java="$prefix/bin/java"
python="$prefix/bin/python"
if [[ $mode == watch ]]; then
    export CUDA_VISIBLE_DEVICES=
    exec "$python" -u "$project/code/minemem/watch_phase0.py"
fi

if [[ $mode == mcu ]]; then
    gpu="${GPU_IDS%,*}"
    report="$(nvidia-smi -i "$gpu" --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits 2>&1)" || {
        status=$?
        printf 'Allocated GPU %s could not be queried. %s\n' "$gpu" "$report" >&2
        exit "$status"
    }
    IFS=, read -r gpu_uuid gpu_name gpu_mib <<< "$report"
    if [[ $gpu_uuid != GPU-* || $gpu_name != *A100* || $gpu_name != *40GB* ||
          ! ${gpu_mib//[[:space:]]/} =~ ^[0-9]+$ ]] ||
       (( gpu_mib < 38000 || gpu_mib > 42000 )); then
        printf 'Expected A100-40GB for allocated GPU %s. Found %s\n' "$gpu" "$report" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$gpu_uuid"
else
    export CUDA_VISIBLE_DEVICES=
fi

if [[ ! -x $java || ! -x $python ]]; then
    printf 'Install Python and JDK 8 in %s first.\n' "$prefix" >&2
    exit 1
fi
if ! "$java" -version 2>&1 | awk '/version "1\.8\./ { found=1 } END { exit !found }'; then
    printf 'The simulator requires JDK 8 at %s.\n' "$java" >&2
    exit 1
fi
for executable in Xvfb xvfb-run xauth; do
    if ! command -v "$executable" >/dev/null; then
        printf 'Missing headless rendering executable: %s\n' "$executable" >&2
        exit 1
    fi
done
if [[ ! -f $MINESTUDIO_DIR/engine/build/libs/mcprec-6.13.jar ]]; then
    printf 'Install the MineStudio engine in %s first.\n' "$MINESTUDIO_DIR" >&2
    exit 1
fi

mkdir -p -- "$TMPDIR" "$WORK/results" "$WORK/runtime/java-home" "$XDG_CACHE_HOME"
export JAVA_HOME="$prefix" PATH="$prefix/bin:$PATH"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR'"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$WORK/code/JarvisVLA"
cd -- "$WORK"
if [[ $mode == mcu && ${MINEMEM_REQUIRE_IDLE:-0} == 1 ]]; then
    "$python" "$project/code/minemem/watch_phase0.py" --check-idle "$gpu" || exit 75
fi
if [[ $mode == mcu ]]; then
    export MINEMEM_ACTIVE_GPU="$gpu"
    printf 'MCU GPU: %s\n' "$gpu"
fi
out="$(mktemp -d "$WORK/results/phase0-${mode}-XXXXXXXX")"
printf 'Run output: %s\n' "$out"
if [[ $mode == capacity ]]; then
    exec "$python" -u "$project/code/minemem/phase0_capacity.py" "$out" "${@:2}"
fi
exec xvfb-run -a -e "$out/xvfb.log" "$python" "$project/code/minemem/phase0.py" "$mode" "$out" "${@:2}"
