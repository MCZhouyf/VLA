#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/dart-smoke.status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-dart-smoke-XXXXXXXX)"
bash code/minemem/t1_cpu.sh collect "$run/dart-smoke" 1 --screened --dart --split train --fraction 0.005
