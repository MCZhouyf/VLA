#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/dart-retry.status"' EXIT
for shard in 010 012 016 030 031; do
    test ! -s "$run/dart-data/worker-$shard/episodes.jsonl"
done
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-dart-retry-XXXXXXXX)"
export T1_CPU_SHARDS='10 12 16 30 31' T1_CPU_DISPLAY_BASE=640 T1_CPU_LAUNCH_INTERVAL=3
bash code/minemem/t1_cpu.sh collect "$run/dart-retry" 32 --screened --dart --split train
