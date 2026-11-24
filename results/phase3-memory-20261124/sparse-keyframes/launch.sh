#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
source env.sh
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH="$WORK/code/minemem:$WORK/code/JarvisVLA"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
out="$(cd -- "$(dirname -- "$0")" && pwd -P)"
distance="$1"
shift
trap 'rc=$?; printf "%s
" "$rc" >"$out/d$distance.status"' EXIT
mix="$WORK/results/phase3-control-20261124-xeZzE4MM/mix"
"$WORK/envs/mcvla-phase3/bin/python" -B -u "$out/snapshot/t1_memory_intervention.py" \
  --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B" \
  --training-run "$WORK/results/phase3-steering-paired-2AGguG2e/shared-adapter-probe" \
  --dataset "$mix/dataset.jsonl" --archive-cache "$(cat "$mix/archive_cache")" \
  --distance "$distance" --episodes 1 --offsets 0 --history-stride 16 \
  --output "$out/d$distance.json" "$@"
