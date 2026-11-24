#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
run="$(dirname "$dir")"
trap 'rc=$?; printf "%s\n" "$rc" >"$dir/navigation.status"' EXIT
while [[ ! -s "$run/continuation-no_memory.status" ]]; do sleep 20; done
[[ $(cat "$run/continuation-no_memory.status") == 0 ]]
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-shared-adapter-XXXXXXXX)"
bash code/minemem/t1_exec.sh eval 6 "$dir/nav" --mode nav --training-run "$dir" \
    --seed 2030990000 2030990001 --control-trace
bash code/minemem/t1_exec.sh eval 6 "$dir/nav_offset" --mode nav_offset --training-run "$dir" \
    --seed 2030990100 2030990101 2030990102 2030990103 --control-trace
date -Is
