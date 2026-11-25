#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$run/dual-smoke4.status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-dual-smoke-XXXXXXXX)"
export T1_CPU_DISPLAY_BASE=7092
bash "$run/dual-snapshot/code/minemem/t1_cpu.sh" dual-enclosure "$run/dual-smoke4" 1 --distances 30 --pairs 1 --capture
