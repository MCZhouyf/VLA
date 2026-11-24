#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
base="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$base/smoke.status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-archived-XXXXXXXX)"
export T1_CPU_LAUNCH_INTERVAL=0.8
T1_CPU_SHARDS='0 20 21' T1_CPU_DISPLAY_BASE=6700 bash "$base/snapshot/code/minemem/t1_cpu.sh" pose-replay "$base/smoke" 32 \
  --fixtures "$WORK/results/phase3-speed-xOn8Q5je/screened_fixtures.json" --distances 30 100 \
  --terminal-tolerance 0.1 --archived-outbound-index "$base/outbound-index.csv"
