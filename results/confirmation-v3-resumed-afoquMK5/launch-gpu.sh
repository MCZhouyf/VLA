#!/usr/bin/env bash
set -euo pipefail
run="$1"
extra="$2"
shift 2
original="$(cat "$run/original-run")"
source "$run/frozen-source/env.sh"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$run/frozen-source/code/minemem:$WORK/code/JarvisVLA"
while ! test -s "$run/supplement.status"; do sleep 15; done
test "$(cat "$run/supplement.status")" = 0
printf '%s\n' supplement-audit > "$run/gpu.progress"
status=0
"$WORK/envs/mcvla/bin/python" -B "$run/harness/t1_phase3e_confirmation_cpu.py" freeze "$run/frozen" \
  --plan "$run/seed-plan.json" --inputs "$original/cpu" "$original/cpu-retry" "$run/cpu-input/cpu" || status=$?
test "$status" = 1
python3 - "$run" "$original" <<'PY'
import csv
import json
import sys
from pathlib import Path
run, previous = map(Path,sys.argv[1:])
def scores(path):
    with path.open() as stream:
        return {(row["seed"],row["baseline"]): row for row in csv.DictReader(stream)}
old=scores(previous/"frozen/acceptance.csv")
new=scores(run/"frozen/acceptance.csv")
assert len(old)==714 and len(new)==720
assert all(new[key]==row for key,row in old.items())
supplement={seed for seed,baseline in set(new)-set(old)}
assert supplement=={"1800516419"}
fixtures=json.loads((run/"frozen/fixtures.json").read_text())["fixtures"]
older=json.loads((previous/"frozen/fixtures.json").read_text())["fixtures"]
assert all(fixtures[key]==config for key,config in older.items())
fixed=fixtures["60/1800516419"]
assert fixed["world_seed"]==88407245 and fixed["ground_force_loaded"]
assert "scene_revision" not in fixed and "neutral_gates" not in fixed and "blind_screen" not in fixed
assert len(fixed["walls"])==len(next(v["walls"] for v in older.values() if v["distance"]==60))
metrics=json.loads((run/"frozen/metrics.json").read_text())
assert not metrics["missing"] and not metrics["passed"]
assert metrics["distances"]["30"]["find_structure"]["successes"]==7
assert metrics["distances"]["30"]["scan"]["successes"]==0
print(f"Preserved {len(old)} original scores and added {len(new)-len(old)} for the single preregistered world", flush=True)
PY
"$WORK/envs/mcvla-phase3/bin/python" -B -u "$run/harness/t1_phase3e_confirmation_run.py" --run "$run" --generalization "$extra" "$@"
