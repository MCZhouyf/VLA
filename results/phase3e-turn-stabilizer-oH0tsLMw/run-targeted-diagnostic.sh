#!/usr/bin/env bash
set -euo pipefail
root="$1"
original=$(cat "$root/original-run")
export WORK="${root%/results/*}" TZ=Asia/Shanghai
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-stabilizer-XXXXXXXX)"
trap 'rc=$?; printf "%s\n" "$rc" >"$root/diagnostic.status"' EXIT
printf 'waiting_for_original_gpu7\n' >"$root/diagnostic.progress"
original_results="$original/oracle-gpu7/episodes.csv"
until [[ -s "$original_results" && $(wc -l <"$original_results") -eq 61 ]]; do
    [[ ! -s "$original/stage-a.status" ]]
    sleep 30
done
printf 'waiting_for_gpu7_lock\n' >"$root/diagnostic.progress"
while true; do
    if bash "$root/snapshot/code/minemem/t1_exec.sh" waypoint-eval 7 "$root/diagnostic" \
        --training-run "$original/controller" --fixtures "$original/frozen/fixtures.json" \
        --metrics "$original/frozen/metrics.json" --distances 10 --batch 3 \
        --seeds 2086000000 2086000008 2086000011 --stabilize-turns; then
        break
    else
        status=$?
        [[ $status == 75 ]] || exit "$status"
        sleep 30
    fi
done
printf 'targeted_diagnostic_finished\n' >"$root/diagnostic.progress"
