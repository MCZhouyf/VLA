#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
job="$1"
raw="$2"
first="$3"
retry="$4"
merged="$5"
trap 'printf "%s\n" "$?" > "$job/shortcut_recheck_status"' EXIT
bash code/minemem/t1_cpu.sh acceptance "$retry" 32 --recheck "$raw" --skip-verified "$first" \
   --legacy "$WORK/results/phase1-teacher-Rk9bGTdG" "$WORK/results/phase1-teacher-terrain-retry-oHyVJIwe"
mkdir "$merged"
for attempt in "$first" "$retry"; do
    for dir in "$attempt"/worker-*; do
        path="$merged/worker-$(basename "$attempt")-$(basename "$dir")"
        mkdir "$path"
        cp "$dir/acceptance.csv" "$path/"
        find "$dir" -maxdepth 1 -type f \( -name 'd*-seed*.json' -o -name '*-outbound.csv' \) -exec cp -t "$path" -- {} +
    done
done
envs/mcvla/bin/python code/minemem/t1_v4_report.py "$merged/summary.csv" "$merged"
printf '0\n' >"$merged/exit_status"
