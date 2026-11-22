#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
raw="$1"
recheck="$2"
p="$3"
trap 'printf "%s\n" "$?" > "$p/shortcut_recheck_status"' EXIT
until [[ -s $raw/exit_status ]]; do sleep 15; done
[[ "$(cat "$raw/exit_status")" == 0 ]]
envs/mcvla/bin/python code/minemem/t1_v4_report.py "$raw/unguarded_summary.csv" "$raw" || [[ -s $raw/unguarded_summary.csv ]]
bash code/minemem/t1_cpu.sh acceptance "$recheck" 16 --recheck "$raw" --legacy \
    "$WORK/results/phase1-teacher-Rk9bGTdG" "$WORK/results/phase1-teacher-terrain-retry-oHyVJIwe"
envs/mcvla/bin/python code/minemem/t1_v4_report.py "$recheck/summary.csv" "$recheck"
