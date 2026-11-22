#!/usr/bin/env bash
set -euo pipefail
review="$1"
target="$2"
seeds="$WORK/results/phase1_t1_seeds.json"
fixtures="$WORK/results/t1-respawn-tTWwXxTb/verified-acceptance"
distances=(10 30 60 100)
options=()
scope="$review/confirmed-$target"
scores=("$review"/visibility/worker-*/visibility.csv "$review"/visibility-retry/worker-*/visibility.csv)
if [[ $target == screened ]]; then
    fixtures="$review/screened-d30"
    distances=(30)
    options+=(--candidate --screened-pillar)
    scores=("$review"/screened-d30/worker-*/visibility.csv)
fi
until [[ -s $scope/exit_status ]]; do sleep 30; done
inputs=("$scope"/worker-*/visibility.csv)
if [[ $(cat "$scope/exit_status") != 0 ]]; then
    mapfile -t missing < <(python3 -B - "$seeds" "$scope" "${distances[@]}" <<'PY'
import csv,json,sys
from pathlib import Path
seeds, scope = map(Path, sys.argv[1:3])
wanted={(int(d),n) for d in sys.argv[3:] for n in json.loads(seeds.read_text())["formal"]}
for f in scope.glob("worker-*/visibility.csv"):
    with f.open(newline="") as source:
        for r in csv.DictReader(source):
            wanted.remove((int(r["d"]),int(r["seed"])))
for d,n in sorted(wanted):
    print(d,n)
PY
)
    if (( ${#missing[@]} )); then
        cases=()
        for pair in "${missing[@]}"; do
            read -r d seed <<< "$pair"
            cases+=(--case "$d" "$seed")
        done
        retry="$scope-retry"
        bash code/minemem/t1_cpu.sh visibility "$retry" 12 --visibility-only --fixtures "$fixtures" \
            --seeds-file "$seeds" --distances "${distances[@]}" "${options[@]}" "${cases[@]}"
        inputs+=("$retry"/worker-*/visibility.csv)
    fi
fi
envs/mcvla/bin/python -B code/minemem/t1_shortcut_audit.py summary "$review/confirmed-$target-summary.json" \
    --seeds-file "$seeds" --distances "${distances[@]}" --inputs "${scores[@]}" --teacher-verified "${inputs[@]}"
