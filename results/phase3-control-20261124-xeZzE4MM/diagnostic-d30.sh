#!/usr/bin/env bash
set -euo pipefail
export WORK=/share/project/zhoubowen/zhy/zyf MINEMEM_PROXY=
cd "$WORK"
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/diagnostic-d30.status"' EXIT
base="$WORK/results/phase3-speed-xOn8Q5je"
seeds="$WORK/results/phase1_t1_seeds.json"
export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-diagnostic-XXXXXXXX)"
selected="$(python3 -B - "$seeds" "$base/structure-audit.json" <<'SEEDS'
import json,sys
from pathlib import Path
seeds,audit=(json.loads(Path(p).read_text()) for p in sys.argv[1:])
assert len(seeds["formal"])==len(set(seeds["formal"]))==30
assert all(audit[str(d)]["passed"] for d in (30,60,100))
print(*seeds["formal"])
SEEDS
)"
read -ra frozen <<<"$selected"
for model in no_memory spatial; do
    date -Is
    bash code/minemem/t1_exec.sh eval 7 "$run/diagnostic-d30-$model" \
        --mode diagnostic_d30 --distances 30 --rollout-batch-size 2 \
        --seed "${frozen[@]}" --seeds-file "$seeds" --screened-fixtures "$base/screened_fixtures.json" \
        --training-run "$base/train-$model"
done
date -Is
