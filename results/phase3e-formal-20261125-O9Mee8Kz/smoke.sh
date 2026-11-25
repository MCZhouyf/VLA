#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$run/smoke.status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-phase3e-smoke-XXXXXXXX)"
export T1_CPU_DISPLAY_BASE=7240 T1_CPU_LAUNCH_INTERVAL=1
bash "$run/snapshot/code/minemem/t1_cpu.sh" phase3e "$run/smoke" 4 --seeds 4
