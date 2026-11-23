#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run=/share/project/zhoubowen/zhy/zyf/results/phase3-speed-xOn8Q5je
out=$(cat "$run/structure_formal_path")
until [[ -s "$out.status" ]]; do sleep 45; done
CUDA_VISIBLE_DEVICES= "$WORK/envs/mcvla/bin/python" -B code/minemem/t1_structure_audit.py \
    "$run/structure-audit-initial.json" --seeds-file "$WORK/results/phase1_t1_seeds.json" \
    --inputs "$out"/worker-*/structure.csv
