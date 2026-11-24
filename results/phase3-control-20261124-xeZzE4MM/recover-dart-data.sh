#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd)"
original="$WORK/results/phase3-speed-xOn8Q5je"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/dart-recovery.status"' EXIT
for name in dart-data dart-retry; do
    until [[ -s "$run/$name.status" ]]; do sleep 20; done
done
if [[ $(cat "$run/dart-data.status") == 0 ]]; then
    echo "Original data stage completed and verified the combined quota."
    exit 0
fi
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH="$WORK/code/JarvisVLA"
python="$WORK/envs/mcvla-phase3/bin/python"
if [[ -e "$run/mix" ]]; then mv "$run/mix" "$run/mix-initial-attempt"; fi
source_data="$WORK/data/t1-screened-hzw7nYp7"
"$python" -B code/minemem/t1_control_mix.py \
    --base-index "$source_data/index-full/dataset.jsonl" --base-root "$source_data" \
    --dart-root "$run/dart-data" --dart-root "$run/dart-retry" \
    --frozen "$WORK/results/phase1_t1_seeds.json" --output "$run/mix"
cache="$(mktemp -d /tmp/minemem-t1-dart-recovered-XXXXXXXX)"
"$python" -B code/minemem/t1_stage_archives.py \
    --dataset "$run/mix/dataset.jsonl" --output "$cache" --manifest "$run/mix/archive_sha256.jsonl"
printf '%s\n' "$cache" >"$run/mix/archive_cache"
for name in continuation-no_memory continuation-spatial repair-formal; do
    until [[ -s "$run/$name.status" ]]; do sleep 5; done
    [[ $(cat "$run/$name.status") != 0 ]]
    for kind in status pid log; do
        mv "$run/$name.$kind" "$run/$name.initial-attempt.$kind"
    done
done
for kind in status pid log; do
    mv "$run/dart-data.$kind" "$run/dart-data.initial-attempt.$kind"
done
printf '%s\n' 0 >"$run/dart-data.status"
for variant in no_memory spatial; do
    nohup bash code/minemem/t1_control_continue.sh train "$run" "$original" "$variant" \
        </dev/null >"$run/continuation-$variant.log" 2>&1 &
    printf '%s\n' "$!" >"$run/continuation-$variant.pid"
done
nohup bash code/minemem/t1_control_continue.sh formal "$run" "$original" \
    </dev/null >"$run/repair-formal.log" 2>&1 &
printf '%s\n' "$!" >"$run/repair-formal.pid"
echo "Verified combined trajectories and rearmed continuation queues."
