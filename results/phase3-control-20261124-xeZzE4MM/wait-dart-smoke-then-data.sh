#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/dart-data.status"' EXIT
until [[ -s "$run/dart-smoke.status" ]]; do
    if ! kill -0 "$(cat "$run/dart-smoke.pid")" 2>/dev/null; then
        echo "DART smoke exited without writing a status."
        exit 1
    fi
    sleep 15
done
[[ $(cat "$run/dart-smoke.status") == 0 ]]
exec bash code/minemem/t1_control_continue.sh data "$run" "$WORK/results/phase3-speed-xOn8Q5je"
