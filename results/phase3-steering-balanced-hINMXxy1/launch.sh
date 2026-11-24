#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY= TZ=Asia/Shanghai
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd -P)"
prior="$WORK/results/phase3-control-20261124-xeZzE4MM"
original="$WORK/results/phase3-speed-xOn8Q5je"
export T1_CONTROL_MIX_RUN="$prior" T1_SPATIAL_CONTINUATION_GPU=7
trap 'rc=$?; printf "%s\n" "$rc" >"$run/launch.status"' EXIT
python3 -B - "$prior/mix" "$run" <<'PY'
import hashlib,json,sys
from pathlib import Path
mix,run=map(Path,sys.argv[1:])
dataset=mix/"dataset.jsonl"
digest=hashlib.sha256(dataset.read_bytes()).hexdigest()
assert digest=="89e1e31e455a2524402399c956b3fe076cb2b42d53f6d1cac44fe164af896da4"
cache=Path((mix/"archive_cache").read_text().strip())
rows=list(map(json.loads,(mix/"archive_sha256.jsonl").read_text().splitlines()))
for row in rows:
    path=cache/row["name"]
    assert path.stat().st_size==row["bytes"],path
    with path.open("rb") as stream:
        digest_actual=hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest_actual.update(block)
    assert digest_actual.hexdigest()==row["sha256"],path
payload=dict(dataset=str(dataset),dataset_sha256=digest,cache=str(cache),checked_archives=len(rows),
    train_cap=1000,checkpoint_every=250,validation_every=1000,initial_update=14388,
    optimizer_batch_sources={"original":2,"dart":2},gpu={"no_memory":6,"spatial":7})
(run/"launch-config.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload),flush=True)
PY
for variant in no_memory spatial; do
    nohup bash code/minemem/t1_control_continue.sh train-balanced "$run" "$original" "$variant" \
        >"$run/continuation-$variant.log" 2>&1 </dev/null &
    printf '%s\n' "$!" >"$run/continuation-$variant.pid"
done
nohup bash code/minemem/t1_control_continue.sh formal "$run" "$original" >"$run/repair-formal.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$run/repair-formal.pid"
date -Is
