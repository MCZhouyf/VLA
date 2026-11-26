#!/usr/bin/env bash
# Build both shards of the common Phase B cache without exposing other GPUs.
set -euo pipefail
root="${1:?Supply the Phase B run directory}"
entry="$(dirname -- "${BASH_SOURCE[0]}")/t1_exec.sh"
export WORK="$(dirname "$(dirname "$root")")" TZ=Asia/Shanghai
trap 'rc=$?; printf "%s\n" "$rc" >"$root/cache.status"' EXIT
pids=()
for gpu in 6 7; do
    bash "$entry" waypoint-cache "$gpu" --plan "$root/cache-plan.json" \
        --output "$root/cache" --shard "$((gpu-6))" >>"$root/cache-gpu$gpu.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
python3 - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
plan = json.loads((root / "cache-plan.json").read_text())
seen = {int(item.stem) for item in (root / "cache").glob("*.pt")}
if seen != {row["scene_seed"] for row in plan["episodes"]}:
    raise ValueError("The phase B cache does not cover the frozen plan")
print("Cache seed coverage verified", len(seen), flush=True)
PY

