#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$run/smoke.status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-phase3e-smoke-XXXXXXXX)"
export T1_CPU_DISPLAY_BASE=7470 T1_CPU_LAUNCH_INTERVAL=1
export T1_CPU_SHARDS="0 8 12 21"
bash "$run/snapshot/code/minemem/t1_cpu.sh" phase3e "$run/smoke" 30 --seeds 30 --distances 10 30
