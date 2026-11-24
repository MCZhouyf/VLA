#!/usr/bin/env bash
set -euo pipefail
cd /share/project/zhoubowen/zhy/zyf
run="$(cd -- "$(dirname -- "$0")" && pwd)"
trap 'rc=$?; printf "%s\n" "$rc" >"$run/cpu-initialization-test.status"' EXIT
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 PYTHONPATH="$PWD/code/JarvisVLA" envs/mcvla-phase3/bin/python -B -u \
 code/minemem/test_t1_resume.py --checkpoint "$PWD/models/JarvisVLA-Qwen2-VL-7B"
