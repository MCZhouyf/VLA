#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/gpu-control-probes.status"' EXIT
while [[ ! -s $run/spatial-gates.status ]]; do
  if ! kill -0 "$(cat "$run/spatial-gates.pid")" 2>/dev/null; then
    echo "Spatial gate launcher exited before writing status."
    break
  fi
  sleep 15
done
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-control-probe-XXXXXXXX)"
base="$WORK/results/phase3-speed-xOn8Q5je/train-no_memory"
date -Is
bash code/minemem/t1_exec.sh eval 6 "$run/input-and-history-probes" \
    --training-run "$base" --mode nav_offset --start-probes
date -Is
bash code/minemem/t1_exec.sh eval 6 "$run/no-memory-step-traces" \
    --training-run "$base" --mode nav --control-trace \
    --seed 2030990000 2030990001 2030990002 2030990003 2030990004 2030990008
date -Is
