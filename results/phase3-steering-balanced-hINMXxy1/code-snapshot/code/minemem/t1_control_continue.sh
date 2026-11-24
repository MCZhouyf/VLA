#!/usr/bin/env bash
set -euo pipefail

project="$(cd -- "$(dirname -- "$0")/../.." && pwd -P)"
source "$project/env.sh"
if (( $# < 3 || $# > 4 )); then
    echo 'Usage: t1_control_continue.sh {data|train|train-balanced|formal} CONTROL_RUN ORIGINAL_RUN [no_memory|spatial]' >&2
    exit 2
fi
stage="$1"
run="$2"
original="$3"
[[ "$run" == "$WORK/results/"* && "$original" == "$WORK/results/"* && "$run" != "$original" ]]
repaired="$run/repaired"
source_data="$WORK/data/t1-screened-hzw7nYp7"
seeds="$WORK/results/phase1_t1_seeds.json"
forward="$WORK/results/t1-review-A8vp1K1p/forward-paired/worker-000/forward.csv"
python="$WORK/envs/mcvla-phase3/bin/python"
export HF_HUB_OFFLINE=1 PYTHONPATH="$WORK/code/JarvisVLA"

wait_for_job() {
    local name="$1"
    until [[ -s "$run/$name.status" ]]; do
        if [[ -s "$run/$name.pid" ]] && ! kill -0 "$(cat "$run/$name.pid")" 2>/dev/null; then
            printf 'Required job exited before writing status: %s\n' "$name" >&2
            return 1
        fi
        sleep 20
    done
    [[ $(cat "$run/$name.status") == 0 ]]
}

case "$stage" in
    data)
        [[ $# == 3 ]]
        export CUDA_VISIBLE_DEVICES=
        trap 'rc=$?; printf "%s\n" "$rc" >"$run/dart-data.status"' EXIT
        export T1_LOCAL_DATABASE_ROOT="$(mktemp -d /dev/shm/minemem-t1-dart-XXXXXXXX)"
        bash "$project/code/minemem/t1_cpu.sh" collect "$run/dart-data" 32 --screened --dart --split train
        "$python" -B "$project/code/minemem/t1_control_mix.py" \
            --base-index "$source_data/index-full/dataset.jsonl" \
            --base-root "$source_data" --dart-root "$run/dart-data" \
            --frozen "$seeds" --output "$run/mix"
        cache="$(mktemp -d /tmp/minemem-t1-dart-cache-XXXXXXXX)"
        "$python" -B "$project/code/minemem/t1_stage_archives.py" \
            --dataset "$run/mix/dataset.jsonl" --output "$cache" \
            --manifest "$run/mix/archive_sha256.jsonl"
        printf '%s\n' "$cache" >"$run/mix/archive_cache"
        ;;
    train|train-balanced)
        [[ $# == 4 ]]
        variant="$4"
        mix="$run/mix"
        training_options=()
        case "$variant" in
            no_memory) gpu=6; prerequisite=gpu-control-probes ;;
            spatial)
                gpu="${T1_SPATIAL_CONTINUATION_GPU:-7}"
                case "$gpu" in
                    6) prerequisite=continuation-no_memory ;;
                    7) prerequisite=diagnostic-d30 ;;
                    *) exit 2 ;;
                esac
                ;;
            *) exit 2 ;;
        esac
        trap 'rc=$?; printf "%s\n" "$rc" >"$run/continuation-$variant.status"' EXIT
        if [[ $stage == train-balanced ]]; then
            prior="${T1_CONTROL_MIX_RUN:?Set T1_CONTROL_MIX_RUN to the completed first control run}"
            [[ "$prior" == "$WORK/results/"* && "$prior" != "$run" ]]
            [[ $(cat "$prior/repair-formal.status") == 0 ]]
            mix="$prior/mix"
            [[ $variant != spatial || $gpu == 7 ]]
            training_options+=(--balance-sources --max-updates 1000 --checkpoint-every 250)
        else
            for name in "${T1_CONTROL_DATA_STAGE:-dart-data}" "$prerequisite"; do
                wait_for_job "$name"
            done
        fi
        mkdir -p "$repaired"
        "$python" -B - "$original/train-$variant" "$mix/summary.json" <<'PY'
import json
import sys
from pathlib import Path
source = Path(sys.argv[1])
original = json.loads((source/"summary.json").read_text())
mixed = json.loads(Path(sys.argv[2]).read_text())
assert original["updates"] == 14388 and original["supervised_examples"] == 57549
assert mixed["mixed_train_labels"] == 12000 and mixed["optimizer_updates"] == 3000
assert (source/"adapter/adapter_model.safetensors").is_file()
PY
        bash "$project/code/minemem/t1_exec.sh" train-resilient "$gpu" "$repaired/train-$variant" \
            --dataset "$mix/dataset.jsonl" --archive-cache "$(cat "$mix/archive_cache")" \
            --initialize-from "$original/train-$variant" --variant "$variant" \
            --history-dropout 0.5 --epochs 1 --accumulation 4 --seed 20261122 \
            --learning-rate 0.0001 --cache-vision --validation-batch-size 4 "${training_options[@]}"
        export T1_NAVIGATION_GPU="$gpu"
        bash "$project/code/minemem/t1_screened_eval.sh" gates "$variant" "$repaired" "$forward"
        if [[ $variant == spatial && $gpu == 6 ]]; then
            # The formal coordinator also needs GPU 7, which finishes the original diagnostic.
            wait_for_job diagnostic-d30
        fi
        ;;
    formal)
        [[ $# == 3 ]]
        trap 'rc=$?; printf "%s\n" "$rc" >"$run/repair-formal.status"' EXIT
        wait_for_job continuation-no_memory
        wait_for_job continuation-spatial
        gate_verdict="$(env CUDA_VISIBLE_DEVICES= PYTHONPATH="$project/code/minemem:$PYTHONPATH" \
            "$python" -B - "$repaired" "$forward" <<'PY'
import json
import sys
from pathlib import Path

from t1_trained_eval import gate_rows, validate_aligned_gate, validate_offset_gate

repaired, forward = map(Path, sys.argv[1:])
results = {}
for variant in ("no_memory", "spatial"):
    aligned = repaired/f"eval-{variant}-nav/episodes.csv"
    offset = repaired/f"eval-{variant}-nav_offset/episodes.csv"
    groups = {"aligned": gate_rows(aligned)}
    for cohort in ("in_view", "out_of_view"):
        groups[cohort] = [row for row in gate_rows(offset) if row["cohort"] == cohort]
    result = {cohort: dict(episodes=len(rows), successes=sum(r["success"] == "True" for r in rows))
              for cohort, rows in groups.items()}
    errors = []
    try:
        actual = validate_aligned_gate(aligned, variant)
        if actual != str((repaired/f"train-{variant}").resolve()):
            raise ValueError("Aligned gate refers to a different checkpoint")
    except (SystemExit, ValueError) as error:
        errors.append(str(error))
    try:
        validate_offset_gate(offset, str((repaired/f"train-{variant}").resolve()), forward, variant)
    except (SystemExit, ValueError) as error:
        errors.append(str(error))
    results[variant] = dict(**result, passed=not errors, blockers=errors)
passes = all(value["passed"] for value in results.values())
with (repaired/"gate-decision.json").open("x") as stream:
    json.dump(dict(variants=results, formal_permitted=passes), stream, indent=2)
    stream.write("\n")
print("pass" if passes else "blocked")
PY
)"
        if [[ "$gate_verdict" != pass ]]; then
            echo "Formal evaluation held by the pre-registered navigation gates."
            exit 0
        fi
        cp "$original/screened_fixtures.json" "$original/structure-audit.json" "$repaired/"
        pids=()
        for variant in no_memory spatial; do
            bash "$project/code/minemem/t1_screened_eval.sh" d30 "$variant" "$repaired" "$forward" \
                >"$repaired/formal-d30-$variant.log" 2>&1 &
            pids+=("$!")
        done
        code=0
        for pid in "${pids[@]}"; do
            wait "$pid" || code=1
        done
        [[ $code == 0 ]]
        "$python" -B - "$repaired" "$seeds" <<'PY'
import csv
import json
import sys
from pathlib import Path

repaired, seeds = map(Path, sys.argv[1:])
frozen = set(json.loads(seeds.read_text())["formal"])
successes = {}
for variant in ("no_memory", "spatial"):
    with (repaired/f"eval-{variant}-d30/episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 30 and {int(r["seed"]) for r in rows} == frozen
    assert all(r["mode"] == "formal" and r["model"] == variant and float(r["d"]) == 30 for r in rows)
    successes[variant] = sum(r["success"] == "True" for r in rows)
result = dict(formal_d30_successes=successes, episodes_per_variant=30,
              proceed=successes["no_memory"] < 6 and successes["spatial"] > 15)
with (repaired/"early-decision.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))
PY
        ;;
    *)
        exit 2
        ;;
esac
