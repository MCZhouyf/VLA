#!/usr/bin/env bash
set -euo pipefail

run="$1"
source "$run/frozen-source/env.sh"
prefix="$WORK/envs/mcvla"
canonical="$MINESTUDIO_DIR"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
export JAVA_HOME="$prefix" PATH="$prefix/bin:$PATH"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR' -XX:ActiveProcessorCount=2"
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$run/frozen-source/code/minemem:$WORK/code/JarvisVLA"
mkdir -p "$TMPDIR" "$WORK/runtime/java-home"
(cd "$run/frozen-source" && sha256sum --check --quiet "$run/frozen-code.sha256")
(cd "$run" && sha256sum --check --quiet seed-plan.sha256)

database_root="$(mktemp -d /dev/shm/minemem-t1-confirm-XXXXXXXX)"
workers=32
entry="$run/harness/t1_phase3e_confirmation_cpu.py"
launch() {
    local attempt="$1"
    shift
    local children=()
    for ((i=0; i<workers; i++)); do
        local worker database
        worker="$(printf '%s/worker-%03d' "$attempt" "$i")"
        mkdir -p "$worker/minestudio"
        ln -s "$canonical/engine" "$worker/minestudio/engine"
        database="$(mktemp -d "$database_root/worker-XXXXXXXX")"
        ln -s "$database" "$worker/minestudio/$(hostname)_tmp"
        (
            export MINESTUDIO_DIR="$worker/minestudio" MESA_SHADER_CACHE_DIR="$worker"
            cd "$worker"
            exec xvfb-run -a -n "$((7600+i))" -e "$worker/xvfb.log" "$prefix/bin/python" -B -u \
                "$entry" worker "$worker" --worker "$i" --workers "$workers" --plan "$run/seed-plan.json" "$@"
        ) >"$worker/worker.log" 2>&1 &
        children+=("$!")
        sleep .4
    done
    local status=0
    for child in "${children[@]}"; do
        wait "$child" || status=1
    done
    printf '%s\n' "$status" > "$attempt/exit_status"
}

printf 'cpu_acceptance\n' > "$run/progress"
launch "$run/cpu"
input=("$run/cpu")
# Resume the same preassigned worlds once if MineStudio did not emit all six readings.
complete="$(python3 - "$run/cpu" <<'PY'
import csv,sys
from pathlib import Path
rows = set()
for path in Path(sys.argv[1]).glob("worker-*/acceptance.csv"):
    with path.open() as stream:
        rows.update(row["seed"] for row in csv.DictReader(stream) if row["baseline"] == "start_pose" and not row["error"])
print(len(rows))
PY
)"
if (( complete < 120 )); then
    printf 'cpu_retry_same_seeds\n' > "$run/progress"
    launch "$run/cpu-retry" --retry-from "$run/cpu"
    input+=("$run/cpu-retry")
fi
(cd "$run/frozen-source" && sha256sum --check --quiet "$run/frozen-code.sha256")
if "$prefix/bin/python" -B "$entry" freeze "$run/frozen" --plan "$run/seed-plan.json" --inputs "${input[@]}"; then
    printf 'cpu_passed\n' > "$run/progress"
else
    printf 'cpu_rejected\n' > "$run/progress"
    exit 1
fi
