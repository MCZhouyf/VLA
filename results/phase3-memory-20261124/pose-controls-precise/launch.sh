#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
fixtures="$WORK/results/phase3-speed-xOn8Q5je/screened_fixtures.json"
trap 'rc=$?; printf "%s\n" "$rc" >"$dir/status"' EXIT
while [[ ! -s "$dir/../pose-controls/status" ]]; do sleep 20; done
[[ $(cat "$dir/../pose-controls/status") == 0 ]]
target="$WORK/code/minemem/t1_pose_replay.py"
cmp -s "$dir/../pose-code/code/minemem/t1_pose_replay.py" "$target"
install -m 0644 "$dir/t1_pose_replay.py" "$target.precise"
mv -T "$target.precise" "$target"
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-pose-replay-XXXXXXXX)"
export T1_CPU_LAUNCH_INTERVAL=0.8
T1_CPU_SHARDS=0 T1_CPU_DISPLAY_BASE=6420 bash code/minemem/t1_cpu.sh pose-replay "$dir/smoke" 32 \
    --fixtures "$fixtures" --distances 30 100 --terminal-tolerance 0.1
python3 -B "$dir/collect-summary.py" "$dir" "$fixtures" --smoke
T1_CPU_DISPLAY_BASE=6500 bash code/minemem/t1_cpu.sh pose-replay "$dir/full" 32 \
    --fixtures "$fixtures" --distances 30 100 --terminal-tolerance 0.1 || true
missing="$(python3 -B "$dir/collect-summary.py" "$dir" "$fixtures" --missing)"
if [[ -n "$missing" ]]; then
    printf 'Retrying pose-only shards %s\n' "$missing"
    T1_CPU_SHARDS="$missing" T1_CPU_DISPLAY_BASE=6600 bash code/minemem/t1_cpu.sh pose-replay "$dir/retry" 32 \
        --fixtures "$fixtures" --distances 30 100 --terminal-tolerance 0.1
fi
python3 -B "$dir/collect-summary.py" "$dir" "$fixtures"
date -Is
