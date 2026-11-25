#!/usr/bin/env bash
set -euo pipefail
run="$1"
export WORK="${run%/results/*}"
old=$(cat "$run/provisional-run")
project="$run/snapshot"
source "$project/env.sh"
python="$WORK/envs/mcvla/bin/python"
export PYTHONPATH="$project/code/minemem:$WORK/code/JarvisVLA"
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-phase3e-revised-XXXXXXXX)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/assignment-revision.status"' EXIT
printf 'waiting_for_provisional_cpu\n' >"$run/stage-a.progress"
while [[ ! -f "$run/inspect-provisional-now" ]] && [[ ! -s "$old/acceptance/exit_status" && ! -s "$old/stage-a.status" ]]; do sleep 20; done
"$python" -B - "$old" "$run" <<'PY'
import csv
import json
import pathlib
import runpy
import sys
from collections import Counter
from random import Random
from t1_phase3e import ACCEPTANCE_SEEDS, GOLD_ORDER, collection_gold_slot
old, run = (pathlib.Path(path) for path in sys.argv[1:])
previous = runpy.run_path(str(old/'provisional-code/t1_phase3e.py'))['GOLD_ORDER']
changed = [i for i,(old_slot, slot) in enumerate(zip(previous,GOLD_ORDER,strict=True)) if old_slot!=slot]
missing = []
counts = Counter()
for i,(seed, slot) in enumerate(zip(ACCEPTANCE_SEEDS,GOLD_ORDER,strict=True)):
    rng = Random(seed)
    counts[(rng.randrange(4),rng.sample((0,1),2)[-1],slot)] += 1
    path = old/f'acceptance/worker-{i:03d}/acceptance.csv'
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    if len(rows)!=24 or any(row['error'] for row in rows):
        missing.append(i)
rerun = sorted(set(changed)|set(missing))
audit = dict(assignment_changes=changed, incomplete_provisional=missing, rerun_shards=rerun,
             frozen_slots=dict(zip(ACCEPTANCE_SEEDS,GOLD_ORDER,strict=True)),
             orientation_last_slot_counts={f'{heading}/{last}/{slot}': n for (heading,last,slot),n in sorted(counts.items())},
             majority_by_orientation_and_order=sum(max(counts[heading,last,0], counts[heading,last,1])
                                                   for heading in range(4) for last in range(2)))
(run/'assignment-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(run/'rerun-shards').write_text(' '.join(map(str,rerun))+'\n')
(run/'reuse-shards').write_text(' '.join(str(i) for i in range(30) if i not in rerun)+'\n')
PY
printf 'cpu_acceptance_revised_assignments\n' >"$run/stage-a.progress"
entry="$project/code/minemem/t1_cpu.sh"
T1_CPU_SHARDS="$(cat "$run/rerun-shards")" T1_CPU_DISPLAY_BASE=7650 T1_CPU_LAUNCH_INTERVAL=0.8 \
    bash "$entry" phase3e "$run/acceptance-revised" 30 --seeds 30 || true
retry=$("$python" -B - "$run" <<'PY'
import csv
import pathlib
import sys
run = pathlib.Path(sys.argv[1]); retry = []
for i in map(int, (run/'rerun-shards').read_text().split()):
    path = run/f'acceptance-revised/worker-{i:03d}/acceptance.csv'
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    if len(rows) != 24 or any(row['error'] for row in rows):
        retry.append(i)
print(' '.join(map(str,retry)))
PY
)
if [[ -n $retry ]]; then
    T1_CPU_SHARDS="$retry" T1_CPU_DISPLAY_BASE=7700 T1_CPU_LAUNCH_INTERVAL=0.8 \
        bash "$entry" phase3e "$run/acceptance-retry" 30 --seeds 30
fi
mkdir "$run/acceptance"
for i in $(cat "$run/reuse-shards"); do
    name=$(printf 'worker-%03d' "$i")
    ln -s "$old/acceptance/$name" "$run/acceptance/$name"
done
for i in $(cat "$run/rerun-shards"); do
    name=$(printf 'worker-%03d' "$i")
    if [[ -d "$run/acceptance-retry/$name" ]]; then
        source="$run/acceptance-retry/$name"
    else
        source="$run/acceptance-revised/$name"
    fi
    ln -s "$source" "$run/acceptance/$name"
done
printf '0\n' >"$run/acceptance/exit_status"
bash "$project/code/minemem/t1_phase3e_stage_a.sh" "$run" --reuse-acceptance
