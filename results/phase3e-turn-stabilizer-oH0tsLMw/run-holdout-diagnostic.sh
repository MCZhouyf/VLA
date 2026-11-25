#!/usr/bin/env bash
set -euo pipefail
root="$1"
original=$(cat "$root/original-run")
export WORK="${root%/results/*}" TZ=Asia/Shanghai
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-stabilizer-holdout-XXXXXXXX)"
trap 'rc=$?; printf "%s\n" "$rc" >"$root/holdout.status"' EXIT
printf 'waiting_for_targeted_diagnostic\n' >"$root/holdout.progress"
while [[ ! -s "$root/diagnostic.status" ]]; do sleep 20; done
[[ $(cat "$root/diagnostic.status") == 0 ]]
python3 - "$root/diagnostic" <<'PY'
import csv,json,pathlib,sys
d=pathlib.Path(sys.argv[1]); chosen={2086000000,2086000008,2086000011}
with (d/'episodes.csv').open() as f: results=list(csv.DictReader(f))
assert len(results)==3 and {int(r['seed']) for r in results}==chosen
if not all(r['success']=='True' for r in results): sys.exit('Targeted stabilization did not pass each regression case')
intervened = []
for seed in (2086000000,2086000008):
    with (d/f'd10-seed{seed}-actions.jsonl').open() as f:
        if any(json.loads(row).get('control_override')=='waypoint_turn_stabilizer' for row in f):
            intervened.append(seed)
if not intervened: sys.exit('The two failed reference scenes recovered without exercising the stabilizer')
PY
for arm in original stabilized; do
    printf 'holdout_%s\n' "$arm" >"$root/holdout.progress"
    flags=()
    if [[ $arm == stabilized ]]; then flags+=(--stabilize-turns); fi
    while true; do
        if bash "$root/snapshot/code/minemem/t1_exec.sh" waypoint-eval 7 "$root/holdout-$arm" \
           --training-run "$original/controller" --metrics "$original/frozen/metrics.json" \
           --fixtures "$root/holdout-fixtures.json" --distances 10 --batch 2 \
           --seeds 2116000000 2116010000 2116020000 2116030000 "${flags[@]}"; then
            break
        else
            status=$?
            [[ $status == 75 ]] || exit "$status"
            sleep 30
        fi
    done
done
printf 'holdout_finished\n' >"$root/holdout.progress"
