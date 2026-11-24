#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/spatial-gates.status"' EXIT
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-gates-XXXXXXXX)"
base="$WORK/results/phase3-speed-xOn8Q5je/train-spatial"
for mode in nav nav_offset; do
    date -Is
    bash code/minemem/t1_exec.sh eval 6 "$run/current-spatial-$mode" --mode "$mode" --training-run "$base"
done
date -Is
