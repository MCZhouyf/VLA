#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
result="$1"
verified="$result/verified-acceptance"
frozen="$WORK/results/phase1_t1_seeds.json"
export CUDA_VISIBLE_DEVICES= MPLCONFIGDIR="$WORK/.cache/matplotlib"
trap 'printf "%s\n" "$?" > "$result/pilot_status"' EXIT
source "$result/run_verified_pilot.sh"
