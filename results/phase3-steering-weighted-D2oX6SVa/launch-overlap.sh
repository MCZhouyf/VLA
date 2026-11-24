#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
balanced="$WORK/results/phase3-steering-balanced-hINMXxy1"
export T1_STEERING_INIT_RUN="$balanced" T1_SPATIAL_CONTINUATION_GPU=7
export T1_CONTROL_MIX_RUN="$WORK/results/phase3-control-20261124-xeZzE4MM"
original="$WORK/results/phase3-speed-xOn8Q5je"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/launch.status"' EXIT
printf 'Waiting for a definitive balanced gate result at %s\n' "$balanced"
while true; do
    if verdict="$(python3 -B - "$balanced/repaired" <<'PY'
from pathlib import Path
import csv,json,sys
root=Path(sys.argv[1])
decision=root/"gate-decision.json"
try:
    allowed=json.loads(decision.read_text())["formal_permitted"]
except (FileNotFoundError,ValueError,KeyError):
    allowed=None
if allowed is True:
    print("pass")
    sys.exit()
if allowed is False:
    print("blocked: final gate verdict")
    sys.exit()
for variant in ("no_memory","spatial"):
    for gate,cohort,maximum_failures in (("nav","aligned",2),("nav_offset","in_view",3)):
        path=root/f"eval-{variant}-{gate}/episodes.csv"
        if not path.exists():
            continue
        with path.open() as stream:
            failures={r["seed"] for r in csv.DictReader(stream) if r.get("mode")==gate
                      and r.get("model")==variant and r.get("cohort")==cohort and r.get("success")=="False"}
        if len(failures)>maximum_failures:
            print(f"blocked: {variant} {gate} has {len(failures)} confirmed failures")
            sys.exit()
sys.exit(1)
PY
)"; then
        break
    fi
    if [[ -s "$balanced/repair-formal.status" ]] && [[ $(cat "$balanced/repair-formal.status") != 0 ]]; then
        printf 'Balanced formal check failed before writing a gate decision.\n' >&2
        exit 1
    fi
    sleep 20
done
printf '%s %s\n' "$(date -Is)" "$verdict"
if [[ $verdict == pass ]]; then
    printf 'skipped\n' >"$run/activation.txt"
    exit 0
fi
[[ $(cat "$run/interleaved-candidate/cpu-test.status") == 0 ]]
(cd "$run" && sha256sum -c code-snapshot.sha256)
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
    nohup bash -c '
      balanced="$1"; variant="$2"; run="$3"; original="$4"
      while [[ ! -s "$balanced/continuation-$variant.status" ]]; do sleep 15; done
      if [[ $(cat "$balanced/continuation-$variant.status") != 0 ]]; then
          printf "1\n" >"$run/continuation-$variant.status"
          exit 1
      fi
      exec bash code/minemem/t1_control_continue.sh train-weighted "$run" "$original" "$variant"
    ' _ "$balanced" "$variant" "$run" "$original" \
        >"$run/continuation-$variant.log" 2>&1 </dev/null &
    printf '%s\n' "$!" >"$run/continuation-$variant.pid"
done
nohup bash code/minemem/t1_control_continue.sh formal "$run" "$original" >"$run/repair-formal.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$run/repair-formal.pid"
date -Is
