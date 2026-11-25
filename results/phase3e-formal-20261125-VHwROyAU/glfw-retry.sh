#!/usr/bin/env bash
set -euo pipefail
run="$1"
export WORK="${run%/results/*}"
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-phase3e-glfw-XXXXXXXX)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/glfw-retry.status"' EXIT
T1_CPU_SHARDS=25 T1_CPU_DISPLAY_BASE=7780 bash "$run/snapshot/code/minemem/t1_cpu.sh" phase3e \
  "$run/acceptance-glfw-retry" 30 --seeds 30
python3 - "$run/acceptance-glfw-retry/worker-025/acceptance.csv" <<'PY'
from pathlib import Path
import csv,sys
with Path(sys.argv[1]).open() as report:
    rows=list(csv.DictReader(report))
assert len(rows)==24 and all(not row['error'] and int(row['seed'])==2086000025 for row in rows)
PY
original="$run/acceptance-revised/worker-025"
[[ -d "$original" && ! -L "$original" && $(wc -l <"$original/acceptance.csv") == 1 ]]
mv "$original" "$run/worker-025-glfw-crash"
ln -s "$run/acceptance-glfw-retry/worker-025" "$original"
