#!/usr/bin/env bash
set -euo pipefail
run="$1"
cp "$run"/{supplement-plan.json,frozen-code.sha256,frozen-commit} "$run/cpu-input/" 
ln -s "$run/frozen-source" "$run/cpu-input/frozen-source"
ln -s "$run/harness" "$run/cpu-input/harness"
mv "$run/cpu-input/supplement-plan.json" "$run/cpu-input/seed-plan.json"
(cd "$run/cpu-input" && sha256sum seed-plan.json > seed-plan.sha256)
T1_CONFIRMATION_WORKERS=1 T1_CONFIRMATION_SCENE=ground-repaired T1_CONFIRMATION_DISPLAY_BASE=8420 \
    T1_CONFIRMATION_PILOT=1 bash "$run/harness/t1_phase3e_confirmation_cpu.sh" "$run/cpu-input"
