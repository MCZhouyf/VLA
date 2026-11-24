#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
base="$(cd -- "$(dirname -- "$0")" && pwd -P)"
fixture="$WORK/results/phase3-speed-xOn8Q5je/screened_fixtures.json"
trap 'rc=$?; printf "%s\n" "$rc" >"$base/status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-archived-full-XXXXXXXX)"
export T1_CPU_LAUNCH_INTERVAL=0.8
opts=(--fixtures "$fixture" --distances 30 100 --terminal-tolerance 0.1 --archived-outbound-index "$base/outbound-index.csv")
summary_py="$base/../pose-controls/collect-summary.py"
T1_CPU_DISPLAY_BASE=6800 bash "$base/snapshot/code/minemem/t1_cpu.sh" pose-replay "$base/full" 32 "${opts[@]}" || true
missing="$(python3 -B "$summary_py" "$base" "$fixture" --missing)"
if [[ -n "$missing" ]]; then
    printf 'Retrying archived shards %s\n' "$missing"
    T1_CPU_SHARDS="$missing" T1_CPU_DISPLAY_BASE=6900 bash "$base/snapshot/code/minemem/t1_cpu.sh" pose-replay "$base/retry" 32 "${opts[@]}"
fi
python3 -B "$summary_py" "$base" "$fixture"
