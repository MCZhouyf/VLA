#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
base="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$base/coarse-recovery.status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-archived-coarse-XXXXXXXX)"
T1_CPU_SHARDS=21 T1_CPU_DISPLAY_BASE=6750 bash "$base/snapshot/code/minemem/t1_cpu.sh" pose-replay "$base/coarse-recovery" 32 \
  --fixtures "$WORK/results/phase3-speed-xOn8Q5je/screened_fixtures.json" --distances 100 \
  --terminal-tolerance 0.4 --archived-outbound-index "$base/outbound-index.csv"
