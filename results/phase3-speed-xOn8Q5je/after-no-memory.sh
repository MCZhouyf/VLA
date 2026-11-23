#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run=/share/project/zhoubowen/zhy/zyf/results/phase3-speed-xOn8Q5je
forward=/share/project/zhoubowen/zhy/zyf/results/t1-review-A8vp1K1p/forward-paired/worker-000/forward.csv
trap 'code=$?; printf "%s\n" "$code" >"$run/no_memory_eval.status"' EXIT
until [[ -s "$run/no_memory.status" ]]; do sleep 45; done
[[ $(cat "$run/no_memory.status") == 0 ]]
date -Is
bash code/minemem/t1_screened_eval.sh gates no_memory "$run" "$forward"
if [[ -s "$run/structure-audit.json" ]]; then
    bash code/minemem/t1_screened_eval.sh d30 no_memory "$run" "$forward"
    bash code/minemem/t1_screened_eval.sh remaining no_memory "$run" "$forward"
else
    echo 'Navigation gates complete. A complete structure audit is needed for formal evaluation.'
fi
