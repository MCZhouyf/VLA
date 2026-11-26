#!/usr/bin/env bash
# Verify exact-pose replay on two shards, then collect and audit the full holdouts.
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
root="${1:?Supply the frozen holdout run directory}"
export WORK="$(dirname "$(dirname "$root")")" TZ=Asia/Shanghai
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-swap-XXXXXXXX)"
python="$WORK/envs/mcvla/bin/python"
entry="$project/code/minemem/t1_phase3e_pairs.py"
launcher="$project/code/minemem/t1_cpu.sh"
trap 'rc=$?; printf "%s\n" "$rc" >"$root/collection.status"' EXIT

printf 'paired_smoke\n' > "$root/collection.progress"
"$python" -B - "$root/plan.json" "$root/smoke-plan.json" <<'PY'
import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
plan = json.loads(source.read_text())
plan["cases"] = [case for case in plan["cases"] if case["ordinal"] < 2]
target.write_text(json.dumps(plan, indent=2) + "\n")
PY
missing_shards() {
    "$python" -B - "$@" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text())
completed = {int(path.stem) for attempt in map(Path, sys.argv[2:]) for path in attempt.glob("worker-*/*.zip")}
print(*sorted({row["ordinal"] for row in plan["cases"] if row["seed"] not in completed}))
PY
}

smoke=()
if [[ -d "$root/smoke-verified" ]]; then smoke+=("$root/smoke-verified"); fi
for attempt in 0 1 2; do
    needed="$(missing_shards "$root/smoke-plan.json" "${smoke[@]}")"
    [[ -n "$needed" ]] || break
    output="$root/smoke-resume-$attempt"
    T1_CPU_SHARDS="$needed" T1_CPU_DISPLAY_BASE="$((8160+attempt*10))" \
        bash "$launcher" phase3e-pairs "$output" 2 --plan "$root/plan.json" --limit 2 --previous "${smoke[@]}" \
        > "$root/smoke-resume-$attempt.log" 2>&1 || true
    smoke+=("$output")
done
"$python" -B "$entry" "$root/smoke-index.json" --plan "$root/smoke-plan.json" --audit "${smoke[@]}"

printf 'collecting_frozen_holdouts\n' > "$root/collection.progress"
previous=()
for attempt in 0 1 2; do
    needed="$(missing_shards "$root/plan.json" "${previous[@]}")"
    [[ -n "$needed" ]] || break
    output="$root/holdouts-attempt-$attempt"
    T1_CPU_SHARDS="$needed" T1_CPU_DISPLAY_BASE="$((8200+attempt*50))" T1_CPU_LAUNCH_INTERVAL=.8 \
        bash "$launcher" phase3e-pairs "$output" 30 --plan "$root/plan.json" --previous "${previous[@]}" \
        > "$root/holdouts-attempt-$attempt.log" 2>&1 || true
    previous+=("$output")
done
"$python" -B "$entry" "$root/holdout-index.json" --plan "$root/plan.json" --audit "${previous[@]}"
printf 'holdouts_verified\n' > "$root/collection.progress"
