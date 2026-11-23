#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= CUDA_VISIBLE_DEVICES=
cd "$WORK"
run=/share/project/zhoubowen/zhy/zyf/results/phase3-speed-xOn8Q5je
out=$(cat "$run/structure_formal_path")
seeds="$WORK/results/phase1_t1_seeds.json"
review="$WORK/results/t1-review-A8vp1K1p"
until [[ -s "$out.status" ]]; do sleep 45; done
until [[ -s "$run/structure-recovery-enabled" ]]; do sleep 45; done
early=$(cat "$run/structure_recovery_path" 2>/dev/null || :)
if [[ -n "$early" ]]; then
    until [[ -s "$early.status" ]]; do sleep 20; done
fi
sources=("$out"/worker-*/structure.csv)
if [[ -n "$early" && $(cat "$early.status") == 0 ]]; then
    sources+=("$early"/worker-*/structure.csv)
fi
missing=$(python3 -B - "$seeds" "${sources[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path
expected = {(d, s) for d in (30, 60, 100)
            for s in json.loads(Path(sys.argv[1]).read_text())["formal"]}
seen = set()
for file in sys.argv[2:]:
    with open(file, newline="") as stream:
        seen.update((int(row["d"]), int(row["seed"])) for row in csv.DictReader(stream))
for distance, seed in sorted(expected - seen):
    print(distance, seed)
PY
)
if [[ -n "$missing" ]]; then
    repair=$(mktemp -d "$review/structure-retry-XXXXXXXX")
    printf "%s\n" "$repair" >"$run/structure_additional_path"
    flags=()
    while read -r distance seed; do
        flags+=(--case "$distance" "$seed")
    done <<<"$missing"
    export T1_LOCAL_DATABASE_ROOT=$(mktemp -d /dev/shm/minemem-t1-struct-repair-XXXXXXXX)
    bash code/minemem/t1_cpu.sh visibility "$repair" 4 \
        --structure-scores "$review/screened_acceptance_scores.csv" --seeds-file "$seeds" \
        --distances 30 60 100 "${flags[@]}"
    sources+=("$repair"/worker-*/structure.csv)
fi
envs/mcvla/bin/python -B code/minemem/t1_structure_audit.py "$run/structure-audit-recovered.json" \
    --seeds-file "$seeds" --inputs "${sources[@]}"
python3 -B - "$run" <<'PY'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1])
recovered = run / "structure-audit-recovered.json"
data = json.loads(recovered.read_text())
if all(data[str(d)]["passed"] for d in (30, 60, 100)):
    approved = run / "structure-audit.json"
    if not approved.exists():
        approved.hardlink_to(recovered)
    print("Screened structure gate passed")
else:
    print("Structure gate requires review", recovered)
    sys.exit(11)
PY
