#!/usr/bin/env bash
set -euo pipefail
run=/share/project/zhoubowen/zhy/zyf/results/phase3-speed-xOn8Q5je
first="$run/structure-audit-initial.json"
until [[ -s "$first" ]] && python3 -B -m json.tool "$first" >/dev/null 2>&1; do
    out=$(cat "$run/structure_formal_path")
    if [[ -s "$out.status" && $(cat "$out.status") != 0 ]]; then
        echo 'The structure workers need follow-up before deciding the gate.'
        exit 12
    fi
    sleep 45
done
python3 -B - "$first" "$run/structure-audit.json" <<'PY'
import json
import sys
from pathlib import Path
candidate, accepted = map(Path, sys.argv[1:])
data = json.loads(candidate.read_text())
if not all(data[str(d)]["passed"] for d in (30, 60, 100)):
    print("Structure gate awaits follow-up. Scores remain in", candidate)
    sys.exit(11)
accepted.hardlink_to(candidate)
print("Confirmed structure gate", accepted)
PY
