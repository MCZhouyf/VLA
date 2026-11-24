#!/usr/bin/env bash
set -euo pipefail

export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd)"
original="$WORK/results/phase3-speed-xOn8Q5je"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/dart-freeze.status"' EXIT
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH="$WORK/code/JarvisVLA"
python="$WORK/envs/mcvla-phase3/bin/python"
source_data="$WORK/data/t1-screened-hzw7nYp7"
"$python" -B code/minemem/t1_control_mix.py \
    --base-index "$source_data/index-full/dataset.jsonl" --base-root "$source_data" \
    --dart-root "$run/dart-data" --dart-root "$run/dart-retry" \
    --frozen "$WORK/results/phase1_t1_seeds.json" --output "$run/mix-frozen"
cache="$(mktemp -d /tmp/minemem-t1-dart-frozen-XXXXXXXX)"
"$python" -B code/minemem/t1_stage_archives.py \
    --dataset "$run/mix-frozen/dataset.jsonl" --output "$cache" --manifest "$run/mix-frozen/archive_sha256.jsonl"
printf '%s\n' "$cache" >"$run/mix-frozen/archive_cache"
if [[ -e "$run/mix" ]]; then
    echo "Existing archive coordinator has already built an index. Its lane keeps priority."
    exit 0
fi

python3 -B - "$run" <<'PY_QUEUE'
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
jobs = {
    "dart-recovery": "recover-dart-data.sh",
    "continuation-no_memory": "t1_control_continue.sh train",
    "continuation-spatial": "t1_control_continue.sh train",
    "repair-formal": "t1_control_continue.sh formal",
}
active = {}
for name, expected in jobs.items():
    pid = int((root/f"{name}.pid").read_text())
    proc = Path(f"/proc/{pid}")
    try:
        cmdline = (proc/"cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        continue
    if not cmdline:
        continue
    assert expected in cmdline and str(root) in cmdline, (name, pid)
    children = subprocess.run(["ps", "--no-headers", "-o", "comm", "--ppid", str(pid)],
                              capture_output=True, text=True).stdout.split()
    if set(children) - {"sleep"}:
        raise RuntimeError(f"Queue {name} is executing active work and was preserved")
    active[name] = pid
for pid in active.values():
    os.kill(pid, signal.SIGTERM)


def alive(pid):
    proc = Path(f"/proc/{pid}/stat")
    try:
        return proc.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


limit = time.monotonic() + 23
while any(alive(pid) for pid in active.values()) and time.monotonic() < limit:
    time.sleep(.2)
assert not any(alive(pid) for pid in active.values())
for name in jobs:
    for suffix in ("status", "pid", "log"):
        path = root/f"{name}.{suffix}"
        if path.exists():
            path.rename(root/f"{name}.shard-wait.{suffix}")
(root/"mix").symlink_to("mix-frozen", target_is_directory=True)
(root/"dart-freeze-selection.json").write_text(json.dumps(
    dict(reason="Verified aggregate quota before every shard reached its local quota",
         passive_waiters_superseded=list(active), active_gpu_evaluations_preserved=True,
         source_roots=["dart-data", "dart-retry"]), indent=2)+"\n")
PY_QUEUE

printf '%s\n' 0 >"$run/dart-frozen-ready.status"
export T1_CONTROL_DATA_STAGE=dart-frozen-ready
for variant in no_memory spatial; do
    nohup bash code/minemem/t1_control_continue.sh train "$run" "$original" "$variant" \
        </dev/null >"$run/continuation-$variant.log" 2>&1 &
    printf '%s\n' "$!" >"$run/continuation-$variant.pid"
done
nohup bash code/minemem/t1_control_continue.sh formal "$run" "$original" \
    </dev/null >"$run/repair-formal.log" 2>&1 &
printf '%s\n' "$!" >"$run/repair-formal.pid"
echo "Frozen data adopted and GPU queues active."
