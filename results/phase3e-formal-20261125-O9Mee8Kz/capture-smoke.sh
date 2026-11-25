#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
role="$1"
port="$2"
trap 'rc=$?; printf "%s
" "$rc" >"$run/smoke-$role.status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-phase3e-capture-XXXXXXXX)"
T1_CPU_DISPLAY_BASE="$port" bash "$run/snapshot/code/minemem/t1_cpu.sh" phase3e "$run/smoke-$role" 1 \
  --split "$role" --fraction .0005
