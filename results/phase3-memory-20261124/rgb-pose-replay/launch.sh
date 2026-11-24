#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
out="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$out/status"' EXIT
while [[ ! -s "$out/../pose-controls-archived/status" ]]; do sleep 20; done
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-rgb-pose-XXXXXXXX)"
export T1_CPU_LAUNCH_INTERVAL=0.8
entry="$out/snapshot/code/minemem/t1_cpu.sh"
index="$WORK/data/t1-screened-hzw7nYp7/index-full/dataset.jsonl"
cache="$(cat "$WORK/results/phase3-speed-xOn8Q5je/disk_cache_path")"
T1_CPU_SHARDS=0 T1_CPU_DISPLAY_BASE=7000 bash "$entry" rgb-pose-replay "$out/smoke" 14 \
  --dataset "$index" --archive-cache "$cache"
T1_CPU_SHARDS='1 2 3 4 5 6 7 8 9 10 11 12 13' T1_CPU_DISPLAY_BASE=7020 bash "$entry" rgb-pose-replay "$out/full" 14 \
  --dataset "$index" --archive-cache "$cache"
