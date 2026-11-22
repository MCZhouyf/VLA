#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
raw="$1"
data="$2"
p="$3"
trap 'printf "%s\n" "$?" > "$p/collection_sequential_status"' EXIT
until [[ -s $raw/exit_status ]]; do sleep 15; done
[[ "$(cat "$raw/exit_status")" == 0 ]]
python3 - "$raw" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path
rows = [row for f in Path(sys.argv[1]).glob("worker-*/acceptance.csv")
        for row in csv.DictReader(f.open())]
with_success = [r for r in rows if not r["error"] and float(r["teacher_min_dist"]) <= 3]
assert Counter(r["distance"] for r in rows) == dict.fromkeys(("10", "30", "60", "100"), 50)
assert len({(r["distance"], r["seed"]) for r in rows}) == 200
assert all(n >= 48 for n in Counter(r["distance"] for r in with_success).values())
assert len(with_success) >= 192
PY
frozen="$WORK/results/phase1_t1_seeds.json"
bash code/minemem/t1_cpu.sh collect "$data/train" 32 --split train
envs/mcvla/bin/python code/minemem/t1_data_index.py "$data/index-smoke" "$data/train" --frozen "$frozen" --smoke
for split in validation test; do
    bash code/minemem/t1_cpu.sh collect "$data/$split" 4 --split "$split"
done
envs/mcvla/bin/python code/minemem/t1_data_index.py "$data/index-full" \
    "$data/train" "$data/validation" "$data/test" --frozen "$frozen"
