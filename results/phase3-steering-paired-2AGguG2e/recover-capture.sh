#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/launch.status"' EXIT
source env.sh
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
export PYTHONPATH="$WORK/code/minemem:$WORK/code/JarvisVLA"
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-paired-recovery-XXXXXXXX)"
prior="$WORK/results/phase3-control-20261124-xeZzE4MM"
base="$WORK/data/t1-screened-hzw7nYp7"
mapfile -t old_records < <(find "$base" "$prior" -name episodes.jsonl -path '*/worker-*/*' -type f)
for offset in 100 200 300; do
    source="$run/alternate-$offset"
    if T1_CPU_SHARDS=22 T1_CPU_DISPLAY_BASE="$((5400 + offset))" \
        bash code/minemem/t1_cpu.sh paired-steering "$source" 32 --positions-per-worker 4 --world-offset "$offset"; then
        [[ $(wc -l <"$source/worker-022/episodes.jsonl") == 12 ]]
        ln -s "$source/worker-022" "$run/complete-shards/worker-022"
        break
    fi
done
test -f "$run/complete-shards/worker-022/episodes.jsonl"
"$WORK/envs/mcvla/bin/python" -B code/minemem/t1_paired_steering.py freeze "$run/index" "$run/complete-shards" \
   --existing "$prior/mix/dataset.jsonl" "$base/index-full/dataset.jsonl" "${old_records[@]}" \
   --frozen "$WORK/results/phase1_t1_seeds.json" --expected-positions 128
cache="$(mktemp -d /tmp/minemem-t1-paired-XXXXXXXX)"
"$WORK/envs/mcvla-phase3/bin/python" -B code/minemem/t1_stage_archives.py \
    --dataset "$run/index/dataset.jsonl" --output "$cache" --manifest "$run/index/archive_sha256.jsonl"
printf '%s\n' "$cache" >"$run/index/archive_cache"
date -Is
