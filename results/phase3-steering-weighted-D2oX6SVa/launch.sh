#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
balanced="$WORK/results/phase3-steering-balanced-hINMXxy1"
export T1_STEERING_INIT_RUN="$balanced" T1_SPATIAL_CONTINUATION_GPU=7
export T1_CONTROL_MIX_RUN="$WORK/results/phase3-control-20261124-xeZzE4MM"
original="$WORK/results/phase3-speed-xOn8Q5je"
decision="$balanced/repaired/gate-decision.json"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/launch.status"' EXIT
printf 'waiting for %s\n' "$decision"
until verdict="$(python3 -B - "$decision" <<'PY'
import json,sys
from pathlib import Path
try:
   result=json.loads(Path(sys.argv[1]).read_text())
   assert isinstance(result["formal_permitted"],bool)
except (FileNotFoundError, ValueError, KeyError):
   sys.exit(1)
print("pass" if result["formal_permitted"] else "blocked")
PY
)"; do
    if [[ -s "$balanced/repair-formal.status" ]]; then
        printf 'Balanced formal coordinator stopped before writing a readable gate decision.\n' >&2
        exit 1
    fi
    sleep 20
done
if [[ $verdict == pass ]]; then
    printf 'skipped\n' >"$run/activation.txt"
    printf 'Balanced continuation reached the navigation gates.\n'
    exit 0
fi
[[ $(cat "$balanced/candidate-weighted-control/cpu-test.status") == 0 ]]
for variant in no_memory spatial; do
    [[ $(cat "$balanced/continuation-$variant.status") == 0 ]]
done
for path in code/minemem/t1_train.py code/minemem/t1_control_continue.sh; do
    cmp -s "$balanced/code-snapshot/$path" "$path"
done
for path in code/minemem/t1_train.py code/minemem/t1_control_continue.sh; do
    cp -- "$run/code-snapshot/$path" "$path.stage3-weighted"
    chmod --reference="$path" "$path.stage3-weighted"
    mv -T -- "$path.stage3-weighted" "$path"
done
printf 'started\n' >"$run/activation.txt"
for variant in no_memory spatial; do
    nohup bash code/minemem/t1_control_continue.sh train-weighted "$run" "$original" "$variant" \
        >"$run/continuation-$variant.log" 2>&1 </dev/null &
    printf '%s\n' "$!" >"$run/continuation-$variant.pid"
done
nohup bash code/minemem/t1_control_continue.sh formal "$run" "$original" >"$run/repair-formal.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$run/repair-formal.pid"
date -Is
