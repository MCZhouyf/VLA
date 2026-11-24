#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
stage="$run/training-snapshot/code/minemem"
balanced="$WORK/results/phase3-steering-balanced-hINMXxy1"
weighted="$WORK/results/phase3-steering-weighted-D2oX6SVa"
original="$WORK/results/phase3-speed-xOn8Q5je"
export T1_CONTROL_MIX_RUN="$WORK/results/phase3-control-20261124-xeZzE4MM"
export T1_STEERING_INIT_RUN="$balanced" T1_SPATIAL_CONTINUATION_GPU=7
export T1_PAIRED_MIX_RUN="$run/replay-index"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$WORK/code/minemem:$WORK/code/JarvisVLA"
python="$WORK/envs/mcvla-phase3/bin/python"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/train-launch.status"' EXIT
"$python" -B -u "$stage/test_t1_resume.py" --checkpoint "$WORK/models/JarvisVLA-Qwen2-VL-7B" \
    >"$run/cpu-training-test.log" 2>&1
printf '0\n' >"$run/cpu-training-test.status"
while [[ ! -s "$run/launch.status" ]]; do sleep 20; done
[[ $(cat "$run/launch.status") == 0 ]]
"$python" -B "$stage/t1_paired_steering.py" replay-index "$T1_PAIRED_MIX_RUN" \
  --paired-index "$run/index/dataset.jsonl" --base-index "$T1_CONTROL_MIX_RUN/mix/dataset.jsonl"
cache="$(mktemp -d /tmp/minemem-t1-triplet-replay-XXXXXXXX)"
"$python" -B code/minemem/t1_stage_archives.py --dataset "$T1_PAIRED_MIX_RUN/dataset.jsonl" \
    --output "$cache" --manifest "$T1_PAIRED_MIX_RUN/archive_sha256.jsonl"
printf '%s\n' "$cache" >"$T1_PAIRED_MIX_RUN/archive_cache"
"$python" -B - "$weighted/repaired/eval-no_memory-nav/episodes.csv" <<'PY'
import csv,sys
with open(sys.argv[1]) as stream:
    misses={row["seed"] for row in csv.DictReader(stream)
            if row["cohort"]=="aligned" and row["model"]=="no_memory" and row["success"]=="False"}
if len(misses) < 3:
    raise SystemExit("The weighted navigation gate has not failed")
PY
for path in code/minemem/t1_train.py code/minemem/t1_control_continue.sh; do
    cmp -s "$weighted/code-snapshot/$path" "$path"
done
for name in t1_train.py t1_control_continue.sh t1_screened_eval.sh t1_paired_steering.py; do
    path="code/minemem/$name"
    cp -- "$stage/$name" "$path.stage3-triplets"
    chmod --reference="$path" "$path.stage3-triplets"
    mv -T -- "$path.stage3-triplets" "$path"
done
for variant in no_memory spatial; do
    nohup bash -c '
      previous="$1"; variant="$2"; run="$3"; original="$4"
      while [[ ! -s "$previous/continuation-$variant.status" ]]; do sleep 15; done
      if [[ $(cat "$previous/continuation-$variant.status") != 0 ]]; then
          printf "1\n" >"$run/continuation-$variant.status"
          exit 1
      fi
      exec bash code/minemem/t1_control_continue.sh train-paired "$run" "$original" "$variant"
    ' _ "$weighted" "$variant" "$run" "$original" >"$run/continuation-$variant.log" 2>&1 </dev/null &
    printf '%s\n' "$!" >"$run/continuation-$variant.pid"
done
nohup bash code/minemem/t1_control_continue.sh formal "$run" "$original" >"$run/repair-formal.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$run/repair-formal.pid"
date -Is
