#!/usr/bin/env bash
set -euo pipefail
run="$1"
cd "$WORK"
trap 'printf "%s\n" "$?" >"$run/after-smoke.status"' EXIT
model="$(cat "$run/smoke-no_memory.training-run")"
bash code/minemem/t1_exec.sh eval 6 "$run/navigation-gate" \
    --mode nav --training-run "$model" >"$run/navigation-gate.log" 2>&1
python3 - "$run" <<'PY'
from pathlib import Path
import csv,json,sys
root=Path(sys.argv[1])
rows=list(csv.DictReader((root/'navigation-gate/episodes.csv').open()))
assert len(rows)==10 and {int(r['seed']) for r in rows}==set(range(2030990000,2030990010))
assert all(r['mode']=='nav' and r['model']=='no_memory' and int(r['d'])==5 for r in rows)
successes=sum(r['success']=='True' for r in rows)
result=dict(successes=successes,n=len(rows),required_successes=8,passed=successes>=8)
(root/'navigation-gate/summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if not result['passed']:
    raise SystemExit('No-memory navigation gate did not pass')
PY
