#!/usr/bin/env bash
set -euo pipefail
result="$1"
cd "$WORK"
source env.sh
canonical="$MINESTUDIO_DIR"
prefix="$WORK/envs/mcvla"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH="$WORK/code/JarvisVLA"
export JAVA_HOME="$prefix" PATH="$prefix/bin:$PATH"
export JAVA_TOOL_OPTIONS="-Duser.home='$WORK/runtime/java-home' -Djava.io.tmpdir='$TMPDIR' -XX:ActiveProcessorCount=2"
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$WORK/.cache/matplotlib"
v4="$WORK/results/t1-v4-guarded-K6ARCMSV"
frozen="$WORK/results/phase1_t1_seeds.json"
trap 'printf "%s\n" "$?" > "$result/recovery_status"' EXIT
children=()
for i in 12 18 20 24 30; do
    worker="$(printf '%s/acceptance-retry/worker-%03d' "$result" "$i")"
    mkdir -p "$worker/minestudio"
    ln -s "$canonical/engine" "$worker/minestudio/engine"
    (
        export MINESTUDIO_DIR="$worker/minestudio" MESA_SHADER_CACHE_DIR="$worker"
        cd "$worker"
        exec xvfb-run -a -e "$worker/xvfb.log" "$prefix/bin/python" -B -u \
            "$WORK/code/minemem/t1_respawn.py" acceptance "$worker" --worker "$i" --workers 32 \
            --fixtures "$v4"
    ) >"$worker/worker.log" 2>&1 &
    children+=("$!")
    sleep 12
done
status=0
for child in "${children[@]}"; do wait "$child" || status=1; done
printf '%s\n' "$status" >"$result/acceptance-retry/exit_status"
test "$status" = 0
until [[ -s $result/exit_status ]]; do sleep 25; done
test "$(cat "$result/exit_status")" != 0
mv "$result/exit_status" "$result/initial_status"
verified="$result/verified-acceptance"
mkdir "$verified"
for first in "$result"/acceptance/worker-*; do
    name="$(basename "$first")"
    retry="$result/acceptance-retry/$name"
    if [[ -d $retry ]]; then
        [[ "$(wc -l < "$first/acceptance.csv")" == 1 ]]
        ln -s "$retry" "$verified/$name"
    else
        ln -s "$first" "$verified/$name"
    fi
done
envs/mcvla/bin/python -B code/minemem/t1_respawn_report.py acceptance \
    "$result/acceptance_summary.csv" "$verified"
sed -n '/^children=()/,$p' "$result/run_cpu.sh" \
    | sed 's|"$result/acceptance"|"$verified"|g' >"$result/run_verified_pilot.sh"
source "$result/run_verified_pilot.sh"
