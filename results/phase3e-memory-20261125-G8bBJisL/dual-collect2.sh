#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf TZ=Asia/Shanghai MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s
" "$rc" >"$run/dual-collect2.status"' EXIT
while [[ ! -s "$run/dual-smoke4.status" ]]; do sleep 15; done
[[ $(cat "$run/dual-smoke4.status") == 0 ]]
python3 - "$run/dual-smoke4/worker-000/oracle.csv" <<'PYTHON'
import csv,sys
with open(sys.argv[1]) as source:
    rows=list(csv.DictReader(source))
if len(rows)!=2 or any(row["error"] or row["success"]!="True" for row in rows):
    raise SystemExit(f"Dual-enclosure smoke gate did not pass: {rows}")
PYTHON
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-dual-collect-XXXXXXXX)"
export T1_CPU_LAUNCH_INTERVAL=0.8
entry="$run/dual-snapshot/code/minemem/t1_cpu.sh"
pids=()
for split in train validation test; do
  case "$split" in
    train) n=6; display=7110 ;;
    validation) n=4; display=7140 ;;
    test) n=4; display=7170 ;;
  esac
  T1_CPU_DISPLAY_BASE="$display" bash "$entry" dual-enclosure "$run/dual-data/$split" "$n" \
    --pairs "$n" --capture --split "$split" >"$run/dual-$split.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
