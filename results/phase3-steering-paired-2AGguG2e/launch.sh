#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/launch.status"' EXIT
source env.sh
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
export PYTHONPATH="$WORK/code/minemem:$WORK/code/JarvisVLA"
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-paired-XXXXXXXX)"
export T1_CPU_LAUNCH_INTERVAL=0.8
prior="$WORK/results/phase3-control-20261124-xeZzE4MM"
base="$WORK/data/t1-screened-hzw7nYp7"
python="$WORK/envs/mcvla/bin/python"
mapfile -t old_records < <(find "$base" "$prior" -name episodes.jsonl -path '*/worker-*/*' -type f)
existing=("$prior/mix/dataset.jsonl" "$base/index-full/dataset.jsonl" "${old_records[@]}")
frozen="$WORK/results/phase1_t1_seeds.json"
T1_CPU_SHARDS=0 T1_CPU_DISPLAY_BASE=5140 bash code/minemem/t1_cpu.sh paired-steering "$run/smoke" 32 --positions-per-worker 2
"$python" -B code/minemem/t1_paired_steering.py freeze "$run/smoke-index" "$run/smoke" \
   --existing "${existing[@]}" --frozen "$frozen" --expected-positions 2
T1_CPU_DISPLAY_BASE=5200 bash code/minemem/t1_cpu.sh paired-steering "$run/capture" 32 --positions-per-worker 4 || true
failed=()
mkdir "$run/complete-shards"
for worker in $(seq 0 31); do
    folder="$(printf 'worker-%03d' "$worker")"
    source="$run/capture/$folder"
    if [[ -f $source/episodes.jsonl ]] && [[ $(wc -l <"$source/episodes.jsonl") == 12 ]]; then
        ln -s "$source" "$run/complete-shards/$folder"
    else
        failed+=("$worker")
    fi
done
if (( ${#failed[@]} )); then
    printf 'Retrying incomplete shards: %s\n' "${failed[*]}"
    T1_CPU_SHARDS="${failed[*]}" T1_CPU_DISPLAY_BASE=5300 \
        bash code/minemem/t1_cpu.sh paired-steering "$run/retry" 32 --positions-per-worker 4
    for worker in "${failed[@]}"; do
        folder="$(printf 'worker-%03d' "$worker")"
        ln -s "$run/retry/$folder" "$run/complete-shards/$folder"
    done
fi
"$python" -B code/minemem/t1_paired_steering.py freeze "$run/index" "$run/complete-shards" \
   --existing "${existing[@]}" --frozen "$frozen" --expected-positions 128
cache="$(mktemp -d /tmp/minemem-t1-paired-XXXXXXXX)"
"$WORK/envs/mcvla-phase3/bin/python" -B code/minemem/t1_stage_archives.py \
    --dataset "$run/index/dataset.jsonl" --output "$cache" --manifest "$run/index/archive_sha256.jsonl"
printf '%s\n' "$cache" >"$run/index/archive_cache"
date -Is
