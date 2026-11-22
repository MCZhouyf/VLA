#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$project/env.sh"
export CUDA_VISIBLE_DEVICES=
cd "$WORK"

review="$1"
fixtures="$WORK/results/t1-respawn-tTWwXxTb/verified-acceptance"
seeds="$WORK/results/phase1_t1_seeds.json"
python="$WORK/envs/mcvla/bin/python"
pilot="$review/screened_d30_summary.json"
trap 'printf "%s\n" "$?" >"$review/screened_followup.status"' EXIT
until [[ -s $pilot ]]; do
    if [[ -s $review/screened-d30/exit_status && $(cat "$review/screened-d30/exit_status") != 0 ]]; then
        exit 1
    fi
    sleep 30
done
"$python" -B - "$pilot" <<'PY'
import json, sys
from pathlib import Path
d30 = json.loads(Path(sys.argv[1]).read_text())["30"]
assert d30["n"] == 30
if d30["teacher_success"] < 29 or d30["scan_success"] >= 3 or d30["turn180_success"] >= 3:
    raise SystemExit("The screened 30-block candidate did not clear its CPU checks")
PY

acceptance="$review/screened-acceptance"
if [[ ! -d $acceptance ]]; then
    bash code/minemem/t1_cpu.sh visibility "$acceptance" 20 --candidate --screened-pillar \
        --cohort teacher --fixtures "$fixtures" --seeds-file "$seeds" --skip-teacher-visibility || true
fi
[[ -s $acceptance/exit_status ]]
shopt -s nullglob
inputs=("$acceptance"/worker-*/visibility.csv)
for prior in "$review"/screened-acceptance-retry*; do
    if [[ $prior == *-known-turn || $prior == *-outbound || $prior == *-failed-lanes ]]; then
        until [[ -s $prior/exit_status ]]; do sleep 20; done
    fi
    [[ -f $prior/exit_status ]] || continue
    inputs+=("$prior"/worker-*/visibility.csv)
done
for attempt in 1 2 3; do
    mapfile -t missing < <("$python" -B - "$seeds" "${inputs[@]}" <<'PY'
import csv, json, sys
from pathlib import Path
seeds = json.loads(Path(sys.argv[1]).read_text())["world_seed_by_scenario"]
wanted = {(d,int(s)) for d in (10,30,60,100) for s in seeds}
for path in map(Path,sys.argv[2:]):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            wanted.remove((int(row["d"]),int(row["seed"])))
for d,s in sorted(wanted):
    print(d,s)
PY
)
    if (( ${#missing[@]} == 0 )); then
        break
    fi
    cases=()
    for pair in "${missing[@]}"; do
        read -r d seed <<< "$pair"
        cases+=(--case "$d" "$seed")
    done
    retry="$(mktemp -d "$review/screened-acceptance-retry-XXXXXXXX")"
    bash code/minemem/t1_cpu.sh visibility "$retry" 20 --candidate --screened-pillar \
        --cohort teacher --fixtures "$fixtures" --seeds-file "$seeds" \
        --skip-teacher-visibility "${cases[@]}" || true
    inputs+=("$retry"/worker-*/visibility.csv)
done
"$python" -B code/minemem/t1_shortcut_audit.py summary "$review/screened_acceptance_summary.json" \
    --cohort teacher --seeds-file "$seeds" --inputs "${inputs[@]}"
"$python" -B - "$review/screened_acceptance_summary.json" "$seeds" "${inputs[@]}" <<'PY'
import csv, json, sys
from pathlib import Path

summary, seed_file = map(Path, sys.argv[1:3])
frozen = set(json.loads(seed_file.read_text())["formal"])
for d, row in json.loads(summary.read_text()).items():
    if row["n"]!=50 or row["teacher_success"]<48 or (int(d)>=30 and row["turn180_success"]>=5):
        raise SystemExit("Screened teacher or 180-degree control fell below the acceptance requirement")
counts = {}
for path in map(Path,sys.argv[3:]):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            d = int(row["d"])
            if d>=30 and int(row["seed"]) in frozen:
                counts[d] = counts.get(d,0) + int(row["success"]=="True")
if set(counts)!={30,60,100} or max(counts.values())>=3:
    raise SystemExit(f"Screened RGB scan control failed: {counts}")
print("Screened CPU acceptance passed", counts, flush=True)
PY

data="$(mktemp -d "$WORK/data/t1-screened-XXXXXXXX")"
printf '%s\n' "$data" >"$review/screened_dataset_path.txt"
bash code/minemem/t1_cpu.sh collect "$data/train" 32 --split train --screened
for split in validation test; do
    bash code/minemem/t1_cpu.sh collect "$data/$split" 4 --split "$split" --screened
done
"$python" -B code/minemem/t1_data_index.py "$data/index-smoke" "$data/train" --frozen "$seeds" --smoke --screened
"$python" -B code/minemem/t1_data_index.py "$data/index-full" \
    "$data/train" "$data/validation" "$data/test" --frozen "$seeds" --screened
