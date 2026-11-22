#!/usr/bin/env bash
set -euo pipefail
cd "$WORK"
p="$1"
frozen="$WORK/results/phase1_t1_seeds.json"
export CUDA_VISIBLE_DEVICES= MPLCONFIGDIR="$WORK/.cache/matplotlib"
trap 'printf "%s\n" "$?" > "$p/pilot_report_status"' EXIT
until [[ -s $p/pilot_cpu_status ]]; do sleep 30; done
[[ "$(cat "$p/pilot_cpu_status")" == 0 ]]
envs/mcvla-phase1-vllm/bin/python code/minemem/t1_pilot.py report "$p/cpu-pilot-report" \
   --seeds-file "$frozen" --models random steve1 vpt --inputs "$p"/cpu-*/cliff_T1.csv
until [[ -s $p/pilot_gpu_status ]]; do sleep 30; done
[[ "$(cat "$p/pilot_gpu_status")" == 0 ]]
envs/mcvla-phase1-vllm/bin/python code/minemem/t1_pilot.py report "$p/full-pilot-report" \
   --seeds-file "$frozen" --inputs "$p"/{cpu,jarvis}-*/cliff_T1.csv
