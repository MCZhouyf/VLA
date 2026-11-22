#!/usr/bin/env bash
set -euo pipefail
review="$1"
mode="$2"
seeds="$WORK/results/phase1_t1_seeds.json"
fixtures="$WORK/results/t1-respawn-tTWwXxTb/verified-acceptance"
args=(--fixtures "$fixtures" --seeds-file "$seeds")
if [[ $mode == paired-teacher ]]; then
    bash code/minemem/t1_cpu.sh visibility "$review/teacher-confirmed" 12 "${args[@]}" --teacher-only
elif [[ $mode == paired-forward ]]; then
    bash code/minemem/t1_cpu.sh forward-control "$review/forward-paired" 1
else
    original="$review/$mode"
    until [[ -s $original/exit_status ]]; do sleep 25; done
    distances=(10 30 60 100)
    options=()
    if [[ $mode == candidate-d30 ]]; then
        distances=(30)
        options+=(--candidate)
    fi
    mapfile -t missing < <(python3 -B - "$original" "$seeds" "${distances[@]}" <<'PY'
import sys, csv, json
from pathlib import Path
root, seeds = map(Path, sys.argv[1:3])
wanted = {(int(d),s) for d in sys.argv[3:] for s in json.loads(seeds.read_text())["formal"]}
done = set()
for path in root.glob("worker-*/visibility.csv"):
    with path.open(newline="") as source:
        done.update((int(r["d"]),int(r["seed"])) for r in csv.DictReader(source))
for d, seed in sorted(wanted-done):
    print(d,seed)
PY
)
    inputs=("$original"/worker-*/visibility.csv)
    if (( ${#missing[@]} )); then
        cases=()
        for pair in "${missing[@]}"; do
            read -r d seed <<< "$pair"
            cases+=(--case "$d" "$seed")
        done
        retry="$review/$mode-retry"
        bash code/minemem/t1_cpu.sh visibility "$retry" 12 "${args[@]}" \
            --distances "${distances[@]}" --skip-teacher-visibility "${options[@]}" "${cases[@]}"
        inputs+=("$retry"/worker-*/visibility.csv)
    fi
    if [[ $mode == candidate-d30 ]]; then
        output="$review/candidate_d30_summary.json"
        verified=()
    else
        until [[ -s $review/teacher-confirmed/exit_status ]]; do sleep 25; done
        [[ $(cat "$review/teacher-confirmed/exit_status") == 0 ]]
        output="$review/visibility_summary.json"
        verified=(--teacher-verified "$review"/teacher-confirmed/worker-*/visibility.csv)
    fi
    envs/mcvla/bin/python -B code/minemem/t1_shortcut_audit.py summary "$output" \
        --seeds-file "$seeds" --distances "${distances[@]}" --inputs "${inputs[@]}" "${verified[@]}"
fi
