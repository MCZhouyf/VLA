#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/cpu-aligned-probe.status"' EXIT
env T1_CPU_THREADS=24 bash code/minemem/t1_exec.sh eval-cpu "$run/cpu-aligned-probe" --mode nav \
 --training-run "$WORK/results/phase3-speed-xOn8Q5je/train-no_memory" --start-probes --seed 2030990000
