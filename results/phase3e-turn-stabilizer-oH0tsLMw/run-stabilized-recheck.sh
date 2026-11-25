#!/usr/bin/env bash
set -euo pipefail
root="$1"
original=$(cat "$root/original-run")
export WORK="${root%/results/*}" TZ=Asia/Shanghai
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-stabilized-recheck-XXXXXXXX)"
trap 'rc=$?; printf "%s\n" "$rc" >"$root/recheck.status"' EXIT
printf 'waiting_for_holdout\n' >"$root/recheck.progress"
while [[ ! -s "$root/holdout.status" ]]; do sleep 20; done
[[ $(cat "$root/holdout.status") == 0 ]]
python3 - "$root" "$original" <<'PY'
import csv,json,pathlib,sys
root,original=map(pathlib.Path,sys.argv[1:]); fixture=json.loads((root/'holdout-fixtures.json').read_text())
seeds=set(fixture['formal']); outcomes={}
for arm in ('original','stabilized'):
    with (root/f'holdout-{arm}'/'episodes.csv').open() as source: rows=list(csv.DictReader(source))
    assert len(rows)==4 and all(int(r['d'])==10 for r in rows)
    outcomes[arm]={int(r['seed']):r['success']=='True' for r in rows}
    assert outcomes[arm].keys()==seeds
if sum(outcomes['stabilized'].values())<3 or any(success and not outcomes['stabilized'][seed] for seed,success in outcomes['original'].items()):
    raise SystemExit('Held-out guard diagnostic did not preserve successful cases')
frozen=json.loads((original/'frozen/fixtures.json').read_text())['formal']; assert len(frozen)==len(set(frozen))==30
(root/'recheck-splits.json').write_text(json.dumps({'gpu6':frozen[:15],'gpu7':frozen[15:]},indent=2)+'\n')
PY
app="$root/snapshot/code/minemem"
gpu_job() {
    local gpu="$1" output="$2" status
    shift 2
    while true; do
        if bash "$app/t1_exec.sh" waypoint-eval "$gpu" "$root/$output" \
           --training-run "$original/controller" --metrics "$original/frozen/metrics.json" \
           --fixtures "$original/frozen/fixtures.json" --batch 3 --stabilize-turns "$@"; then
            return 0
        else
            status=$?
            [[ $status == 75 ]] || return "$status"
            sleep 30
        fi
    done
}
finish_wave() {
    local failed=0
    for pid in "$@"; do wait "$pid" || failed=1; done
    [[ $failed == 0 ]]
}
check_gate() {
    python3 - "$root" "$original" "$1" <<'PY'
import csv,json,pathlib,sys
root,original=map(pathlib.Path,sys.argv[1:3]); wave=sys.argv[3]
expected=set(json.loads((original/'frozen/fixtures.json').read_text())['formal']); rows=[]
paths=['recheck-10-gpu6','recheck-10-gpu7']; distances=[10]
if wave=='all': paths+=['recheck-long-gpu6','recheck-long-gpu7']; distances += [30,60,100]
for name in paths:
    with (root/name/'episodes.csv').open() as source: rows.extend(csv.DictReader(source))
assert len(rows)==30*len(distances)
summary={}
for distance in distances:
    part=[row for row in rows if int(row['d'])==distance]; assert len(part)==30 and {int(r['seed']) for r in part}==expected
    count=sum(r['success']=='True' for r in part)
    summary[distance]={'successes':count,'n':30,'rate':count/30}
passed=all(row['successes']>=24 for row in summary.values())
(root/('recheck_oracle_gate.json' if wave=='all' else 'recheck_near_gate.json')).write_text(json.dumps({'passed':passed,'distances':summary},indent=2)+'\n')
if not passed: raise SystemExit(f'The stabilized controller failed the {wave} oracle gate')
PY
}
printf 'stabilized_near_recheck\n' >"$root/recheck.progress"
pids=()
for gpu in 6 7; do
    read -ra seeds < <(python3 -c 'import json,sys; print(*json.load(open(sys.argv[1]))["gpu"+sys.argv[2]])' "$root/recheck-splits.json" "$gpu")
    [[ ${#seeds[@]} == 15 ]]
    gpu_job "$gpu" "recheck-10-gpu$gpu" --distances 10 --seeds "${seeds[@]}" >"$root/recheck-10-gpu$gpu.log" 2>&1 &
    pids+=("$!")
done
finish_wave "${pids[@]}"
check_gate near
printf 'stabilized_long_recheck\n' >"$root/recheck.progress"
gpu_job 6 recheck-long-gpu6 --distances 100 >"$root/recheck-long-gpu6.log" 2>&1 &
a=$!
gpu_job 7 recheck-long-gpu7 --distances 30 60 >"$root/recheck-long-gpu7.log" 2>&1 &
b=$!
finish_wave "$a" "$b"
check_gate all
printf 'stabilized_oracle_gate_passed\n' >"$root/recheck.progress"
