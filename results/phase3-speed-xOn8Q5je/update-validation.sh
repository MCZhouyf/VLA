#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
cd /share/project/zhoubowen/zhy/zyf
run=/share/project/zhoubowen/zhy/zyf/results/phase3-speed-xOn8Q5je
prior=
while :; do
    vals="$run/train-no_memory/val_loss.csv"
    if [[ -s "$vals" && $(wc -l <"$vals") -ge 2 ]]; then
        revision=$(sha256sum "$vals")
        if [[ $revision != "$prior" ]]; then
            envs/mcvla-phase1-vllm/bin/python -B code/minemem/t1_validation_plot.py \
                "$run/val_loss.svg" --full "$run"
            prior="$revision"
        fi
    fi
    [[ ! -s "$run/no_memory.status" ]] || exit 0
    sleep 45
done
