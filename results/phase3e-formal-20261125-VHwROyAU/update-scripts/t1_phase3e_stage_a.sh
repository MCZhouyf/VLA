#!/usr/bin/env bash
# Gate data collection, waypoint training and oracle evaluation on the frozen v3 protocol.
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
run="${1:?Supply the phase 3e run directory}"
[[ $# == 1 || ( $# == 2 && $2 == --reuse-acceptance ) ]]
export TZ=Asia/Shanghai
source "$project/env.sh"
cd "$WORK"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/stage-a.status"' EXIT
while [[ ! -s "$run/smoke.status" ]]; do sleep 20; done
[[ $(cat "$run/smoke.status") == 0 ]]
python="$WORK/envs/mcvla/bin/python"
export PYTHONPATH="$project/code/minemem:$WORK/code/JarvisVLA"
"$python" -B - "$run/smoke" <<'PY'
import csv
import sys
from pathlib import Path

reports = sorted(Path(sys.argv[1]).glob('worker-*/acceptance.csv'))
rows = []
for report in reports:
    with report.open() as source:
        rows.extend(csv.DictReader(source))
teachers = [row for row in rows if row['baseline'] == 'teacher']
if not teachers or any(row['error'] for row in rows) or any(row['success'] != 'True' for row in teachers):
    raise SystemExit('Resolve the teacher or scene errors in the CPU smoke test first')
PY
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-phase3e-XXXXXXXX)"
export T1_CPU_LAUNCH_INTERVAL=0.8
entry="$project/code/minemem/t1_cpu.sh"
exec_gpu="$project/code/minemem/t1_exec.sh"

gpu_job() {
    local mode="$1" gpu="$2" code
    shift 2
    while true; do
        if bash "$exec_gpu" "$mode" "$gpu" "$@"; then return 0; else code=$?; fi
        if [[ $code != 75 ]]; then return "$code"; fi
        printf '%s GPU %s busy, checking again in 60 seconds\n' "$(date -Iseconds)" "$gpu"
        sleep 60
    done
}

printf 'cpu_acceptance\n' >"$run/stage-a.progress"
if [[ $# == 1 ]]; then
    T1_CPU_DISPLAY_BASE=7300 bash "$entry" phase3e "$run/acceptance" 30 --seeds 30
else
    [[ $(cat "$run/acceptance/exit_status") == 0 ]]
fi
"$python" -B "$project/code/minemem/t1_phase3e_index.py" acceptance \
  --input "$run/acceptance" --output "$run/frozen"

printf 'collection\n' >"$run/stage-a.progress"
pids=()
for split in train validation test; do
    case "$split" in
        train) workers=24; display=7350 ;;
        validation) workers=4; display=7380 ;;
        test) workers=4; display=7390 ;;
    esac
    T1_CPU_DISPLAY_BASE="$display" bash "$entry" phase3e "$run/data/$split" "$workers" --split "$split" \
      >"$run/collect-$split.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[[ $failed == 0 ]]
T1_CPU_DISPLAY_BASE=7410 bash "$entry" phase3e "$run/data/dart" 32 --split dart >"$run/collect-dart.log" 2>&1

printf 'archive_verification\n' >"$run/stage-a.progress"
"$python" -B "$project/code/minemem/t1_phase3e_index.py" index \
  --collection "$run/data" --frozen "$run/frozen/fixtures.json" --output "$run/index" \
  --previous "$WORK/data/t1-screened-hzw7nYp7/index-full/dataset.jsonl" \
    "$WORK/results/phase3-control-20261124-xeZzE4MM/mix/dataset.jsonl" \
  --previous-fixtures "$WORK/results/phase3-speed-xOn8Q5je/screened_fixtures.json"
cache="$(mktemp -d /tmp/minemem-phase3e-XXXXXXXX)"
printf '%s\n' "$cache" >"$run/archive_cache"
"$python" -B "$project/code/minemem/t1_stage_archives.py" \
  --dataset "$run/index/dataset.jsonl" --output "$cache" --workers 8 --manifest "$run/archive_sha256.jsonl"

printf 'gpu6_controller_training\n' >"$run/stage-a.progress"
gpu_job train-resilient 6 "$run/controller" \
  --dataset "$run/index/dataset.jsonl" --archive-cache "$cache" --variant no_memory \
  --initialize-from "$WORK/results/phase3-control-20261124-xeZzE4MM/repaired/train-no_memory" \
  --history-dropout .5 --cache-vision --epochs 1 --accumulation 4 --train-microbatch-size 2 \
  --validation-batch-size 4 --learning-rate .00005 --max-updates 8000 >"$run/controller.log" 2>&1

printf 'oracle_waypoints\n' >"$run/stage-a.progress"
pids=()
for gpu in 6 7; do
    if [[ $gpu == 6 ]]; then ds=(10 100); else ds=(30 60); fi
    gpu_job waypoint-eval "$gpu" "$run/oracle-gpu$gpu" --training-run "$run/controller" \
      --metrics "$run/frozen/metrics.json" --fixtures "$run/frozen/fixtures.json" \
      --distances "${ds[@]}" --batch 3 >"$run/oracle-gpu$gpu.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[[ $failed == 0 ]]
"$python" -B - "$run" <<'PY'
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
frozen = set(json.loads((run/'frozen/fixtures.json').read_text())['formal'])
rows = []
for gpu in (6, 7):
    with (run/f'oracle-gpu{gpu}/episodes.csv').open() as source:
        rows.extend(csv.DictReader(source))
result = {}
for d in (10, 30, 60, 100):
    cases = [row for row in rows if int(row['d']) == d]
    if len(cases) != 30 or {int(row['seed']) for row in cases} != frozen:
        raise ValueError(f'Incomplete frozen oracle coverage for {d}')
    wins = sum(row['success'] == 'True' for row in cases)
    result[d] = dict(n=len(cases), successes=wins, rate=wins/len(cases))
passed = all(row['rate'] >= .80 for row in result.values())
(run/'oracle_gate.json').write_text(json.dumps(dict(passed=passed, distances=result), indent=2)+'\n')
if not passed:
    raise SystemExit('The waypoint controller has not met the four-distance oracle gate')
PY
printf 'oracle_gate_passed\n' >"$run/stage-a.progress"
