#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$project/env.sh"
export CUDA_VISIBLE_DEVICES=
prefix="$WORK/envs/mcvla"
venv="$WORK/envs/mcvla-phase1-vllm"
checkpoint="$WORK/models/JarvisVLA-Qwen2-VL-7B"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$WORK/code/JarvisVLA"
export JAVA_HOME="$prefix" PATH="$prefix/bin:$PATH"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR'"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export JAVA_TOOL_OPTIONS="$JAVA_TOOL_OPTIONS -XX:ActiveProcessorCount=2"
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$WORK/runtime/java-home"
cd "$WORK"
launcher="$project/code/minemem/phase1_jarvis.sh"

bind_allocated_gpu() {
    local gpu="$1"
    local expected_bus
    "$prefix/bin/python" "$project/code/minemem/watch_phase0.py" --check-idle "$gpu"
    expected_bus="$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader)"
    # vLLM 0.7.3's NVML adapter requires a numeric CUDA visibility mask.
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$venv/bin/python" - "$expected_bus" <<'PY'
import ctypes
import sys

driver = ctypes.CDLL("libcuda.so.1")
device = ctypes.c_int()
bus = ctypes.create_string_buffer(32)
if (driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), 0)
        or driver.cuDeviceGetPCIBusId(bus, len(bus), device)):
    raise SystemExit("Could not identify the allocated CUDA device")

def pci_address(value):
    return tuple(int(part, 16) for part in value.replace(".", ":").split(":"))

if pci_address(bus.value.decode()) != pci_address(sys.argv[1]):
    raise SystemExit("CUDA visibility does not match the requested GPU's PCI address")
PY
}

serve() {
    local port="$1" memory="$2" max_length="$3" sequences="$4" images="$5"
    exec "$venv/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "$checkpoint" --dtype bfloat16 \
        --host 127.0.0.1 --port "$port" --enforce-eager \
        --gpu-memory-utilization "$memory" --max-model-len "$max_length" \
        --max-num-seqs "$sequences" --limit-mm-per-prompt "image=$images" --disable-log-requests
}

await_service() {
    local port="$1"
    local deadline=$((SECONDS + 1200))
    until curl -fs --max-time 2 "http://127.0.0.1:$port/health"; do
        if (( SECONDS >= deadline )); then
            printf 'Inference service on %s did not become ready\n' "$port" >&2
            return 1
        fi
        sleep 4
    done
}

isolate_simulator() {
    local output="$1"
    [[ $output == "$WORK/results/"* ]]
    mkdir -p "$output/minestudio"
    ln -s "$MINESTUDIO_DIR/engine" "$output/minestudio/engine"
    export MINESTUDIO_DIR="$output/minestudio" MESA_SHADER_CACHE_DIR="$output"
    cd "$output"
}

check_seeds() {
    local task="$1" label="$2" port="$3" offset="$4" stride="$5"
    local index seed directory status=0
    for ((index=offset; index<10; index+=stride)); do
        seed=$((19961103 + index))
        directory="$output/$label-$seed"
        bash "$launcher" check "$task" "$directory" --base-url "http://127.0.0.1:$port/v1" \
            --start-seed "$seed" --count 1 >"$directory.log" 2>&1 || status=1
    done
    return "$status"
}

case "${1:-}" in
    serve)
        if [[ $# != 3 || ! $2 =~ ^[67]$ || ! $3 =~ ^[0-9]+$ ]]; then
            echo 'Usage: phase1_jarvis.sh serve {6|7} PORT' >&2
            exit 2
        fi
        bind_allocated_gpu "$2"
        serve "$3" 0.85 4096 4 5
        ;;
    pair)
        if [[ $# != 5 || ! $2 =~ ^[67]$ || ! $3 =~ ^[0-9]+$ || ! $4 =~ ^[0-9]+$ ||
              $3 == "$4" || $5 != "$WORK/results/"* ]]; then
            echo 'Usage: phase1_jarvis.sh pair {6|7} PORT1 PORT2 OUTPUT' >&2
            exit 2
        fi
        gpu="$2" first_port="$3" second_port="$4" output="$5"
        mkdir -p "$output"
        exec 9>"$TMPDIR/phase1-vllm-gpu-$gpu.lock"
        flock -n 9
        bind_allocated_gpu "$gpu"
        trap 'for child in $(jobs -pr); do kill "$child"; done' EXIT
        trap 'exit 143' TERM INT
        serve "$first_port" 0.47 3072 1 3 >"$output/$first_port.log" 2>&1 &
        first_child=$!
        await_service "$first_port"
        kill -0 "$first_child"
        first_apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"
        if [[ ! $first_apps =~ ^[0-9]+$ ]]; then
            echo 'Expected one compute process before starting the second instance' >&2
            exit 1
        fi
        serve "$second_port" 0.47 3072 1 3 >"$output/$second_port.log" 2>&1 &
        second_child=$!
        await_service "$second_port"
        kill -0 "$second_child"
        nvidia-smi -i "$gpu" --query-gpu=index,memory.used,memory.total --format=csv >"$output/gpu.csv"
        nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv >"$output/instances.csv"
        if [[ $(sed -n '2,$p' "$output/instances.csv" | wc -l) != 2 ]] ||
            ! grep -q "^$first_apps," "$output/instances.csv"; then
            echo 'GPU compute process membership changed during startup' >&2
            exit 1
        fi
        printf 'Two instances ready on GPU %s\n' "$gpu"
        wait -n "$first_child" "$second_child"
        ;;
    check)
        shift
        if [[ $# -lt 2 || $2 != "$WORK/results/"* ]]; then
            echo 'Provide an absolute output directory under $WORK/results' >&2
            exit 2
        fi
        if [[ $1 != navigation && -d $WORK/code/MineStudio-muyao-72cdab3/minestudio ]]; then
            export PYTHONPATH="$WORK/code/MineStudio-muyao-72cdab3:$PYTHONPATH"
        fi
        isolate_simulator "$2"
        exec xvfb-run -a "$venv/bin/python" -u "$project/code/minemem/phase1_jarvis_gate.py" \
            --checkpoint "$checkpoint" "$@"
        ;;
    t1)
        if [[ $# -lt 2 ]]; then
            echo 'Usage: phase1_jarvis.sh t1 OUTPUT ...' >&2
            exit 2
        fi
        output="$2"
        shift 2
        isolate_simulator "$output"
        exec xvfb-run -a "$venv/bin/python" -u "$project/code/minemem/phase1_t1_eval.py" \
            --work "$WORK" --output "$output" "$@"
        ;;
    batch)
        if [[ $# != 4 || $2 != "$WORK/results/"* ]]; then
            echo 'Usage: phase1_jarvis.sh batch OUTPUT PORT6 PORT7' >&2
            exit 2
        fi
        output="$2"
        mkdir -p "$output"
        for port in "$3" "$4"; do
            [[ $port =~ ^[0-9]+$ ]]
            await_service "$port"
        done
        workers=()
        for offset in 0 1; do
            check_seeds craft/craft_crafting_table craft "$3" "$offset" 2 &
            workers+=("$!")
            check_seeds smelt/iron_ingot smelt "$4" "$offset" 2 &
            workers+=("$!")
            check_seeds navigation navigation "$3" "$offset" 2 &
            workers+=("$!")
        done
        printf 'Gate rollout workers %s\n' "${workers[*]}"
        status=0
        for worker in "${workers[@]}"; do
            wait "$worker" || status=1
        done
        exit "$status"
        ;;
    *)
        echo 'Usage: phase1_jarvis.sh {serve GPU PORT|pair GPU PORT1 PORT2 OUTPUT|check TASK OUTPUT ...|batch OUTPUT PORT6 PORT7|t1 OUTPUT ...}' >&2
        exit 2
        ;;
esac
