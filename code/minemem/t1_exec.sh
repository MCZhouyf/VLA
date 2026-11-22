#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$project/env.sh"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
export PYTHONPATH="$WORK/code/JarvisVLA"
prefix="$WORK/envs/mcvla"
export JAVA_HOME="$prefix" PATH="$prefix/bin:$PATH"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR' -XX:ActiveProcessorCount=2"
export LP_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export LIBGL_ALWAYS_SOFTWARE=1
mkdir -p "$TMPDIR" "$WORK/runtime/java-home"
cd "$WORK"

bind_gpu() {
    [[ $1 =~ ^[67]$ ]]
    exec 9>"$TMPDIR/phase1-vllm-gpu-$1.lock"
    flock -w 30 9
    local waits=0
    until "$prefix/bin/python" "$project/code/minemem/watch_phase0.py" --check-idle "$1"; do
        # A CUDA subprocess can exit before its utilization sample clears.
        waits=$((waits + 1))
        if ((waits >= 15)); then return 75; fi
        sleep 2
    done
    CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)"
    [[ $CUDA_VISIBLE_DEVICES == GPU-* ]]
    export CUDA_VISIBLE_DEVICES
}

simulator_dir() {
    [[ $1 == "$WORK/results/"* ]]
    mkdir -p "$1/minestudio"
    ln -s "$MINESTUDIO_DIR/engine" "$1/minestudio/engine"
    export MINESTUDIO_DIR="$1/minestudio" MESA_SHADER_CACHE_DIR="$1"
    cd "$1"
}

case "${1:-}" in
    pilot)
        output="$2"
        shift 2
        simulator_dir "$output"
        exec xvfb-run -a -e "$output/xvfb.log" "$WORK/envs/mcvla-phase1-vllm/bin/python" -B -u \
            "$project/code/minemem/t1_pilot.py" run --work "$WORK" --output "$output" "$@"
        ;;
    protected-pilot|historical-pilot|death-replay)
        mode="$1"
        output="$2"
        shift 2
        simulator_dir "$output"
        options=(pilot "$output" --work "$WORK")
        case "$mode" in
            historical-pilot) options+=(--historical) ;;
            death-replay) options=(death-replay "$output") ;;
        esac
        exec xvfb-run -a -e "$output/xvfb.log" "$WORK/envs/mcvla-phase1-vllm/bin/python" -B -u \
            "$project/code/minemem/t1_respawn.py" "${options[@]}" "$@"
        ;;
    train)
        gpu="$2"
        shift 2
        bind_gpu "$gpu"
        exec "$WORK/envs/mcvla-phase3/bin/python" -B -u \
            "$project/code/minemem/t1_train.py" --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B" "$@"
        ;;
    eval)
        gpu="$2"
        output="$3"
        shift 3
        bind_gpu "$gpu"
        simulator_dir "$output"
        exec xvfb-run -a -e "$output/xvfb.log" "$WORK/envs/mcvla-phase3/bin/python" -B -u \
            "$project/code/minemem/t1_trained_eval.py" --output "$output" \
            --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B" "$@"
        ;;
    *)
        echo 'Usage: t1_exec.sh {pilot|protected-pilot|historical-pilot|death-replay} OUTPUT or {train GPU|eval GPU OUTPUT} [options]' >&2
        exit 2
        ;;
esac
