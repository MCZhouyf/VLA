#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
fixtures="$WORK/results/phase3-speed-xOn8Q5je/screened_fixtures.json"
trap 'rc=$?; printf "%s\n" "$rc" >"$dir/status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-pose-replay-XXXXXXXX)"
export T1_CPU_LAUNCH_INTERVAL=0.8
T1_CPU_SHARDS=0 T1_CPU_DISPLAY_BASE=6120 bash code/minemem/t1_cpu.sh pose-replay "$dir/smoke" 32 \
    --fixtures "$fixtures" --distances 30 100
python3 -B "$dir/collect-summary.py" "$dir" "$fixtures" --smoke
T1_CPU_DISPLAY_BASE=6200 bash code/minemem/t1_cpu.sh pose-replay "$dir/full" 32 \
    --fixtures "$fixtures" --distances 30 100 || true
missing="$(python3 -B "$dir/collect-summary.py" "$dir" "$fixtures" --missing)"
if [[ -n "$missing" ]]; then
    printf 'Retrying pose-only shards %s\n' "$missing"
    T1_CPU_SHARDS="$missing" T1_CPU_DISPLAY_BASE=6300 bash code/minemem/t1_cpu.sh pose-replay "$dir/retry" 32 \
        --fixtures "$fixtures" --distances 30 100
fi
python3 -B "$dir/collect-summary.py" "$dir" "$fixtures"
date -Is
