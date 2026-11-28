#!/usr/bin/env bash
set -euo pipefail
run="$1"
controller="$2"
offline="$3"
export WORK="$(realpath "$run/../../..")" TZ=Asia/Shanghai CUDA_VISIBLE_DEVICES=
source "$run/frozen-source/env.sh"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$run/frozen-source/code/minemem:$WORK/code/JarvisVLA:$run/recovery-harness"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$WORK/envs/mcvla-phase3/bin/python" -B - <<'PY'
from types import SimpleNamespace
import numpy as np
from minestudio.simulator.entry import MinecraftSim, CameraConfig, CameraHierarchicalMapping, ActionTransformer
from phase1_t1 import CAMERA
from t1_phase3e_confirmation_gpu import drive_script
from t1_trained_eval import ControlTrace

config=CameraConfig(**CAMERA)
sim=MinecraftSim.__new__(MinecraftSim)
sim.action_mapper=CameraHierarchicalMapping(n_camera_bins=config.n_camera_bins)
sim.action_transformer=ActionTransformer(**config.action_transformer_kwargs)
sim.noop_action=lambda: {"camera":np.zeros(2)}
trace=ControlTrace(None,None)
def turn():
    agent,complete=yield
    action=sim.agent_action_to_env_action(agent)
    assert complete and action["forward"]==1 and action["camera"].shape==(2,)
    assert not np.any(action["camera"])
    trace.record_environment(action)
    return int(action["forward"])
assert drive_script(turn(),SimpleNamespace(sim=sim),trace)==1
assert trace.controls[0]["buttons"]["forward"]==1
print("B2 agent action decodes to one forward environment action",flush=True)
PY
mapfile -t seeds < <(python3 - "$run/seed-plan.json" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["cases"]:
    if row["distance"]==30: print(row["seed"])
PY
)
(( ${#seeds[@]} == 30 ))
trial_tmp="$(mktemp -d /dev/shm/minemem-t1-confirm-7-XXXXXX)"
export T1_LOCAL_DATABASE_ROOT="$trial_tmp"
trap 'rmdir "$trial_tmp" 2>/dev/null || true' EXIT
exec bash "$run/recovery-harness/t1_phase3e_confirmation_gpu.sh" "$run" 7 "$run/eval/B/b2/attempt-3" \
   --mode b2 --training-run "$controller" --room-weights "$offline/room-head.pt" \
   --fixtures "$run/frozen/fixtures.json" --batch 3 --seeds "${seeds[@]}"
