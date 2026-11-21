#!/usr/bin/env bash
# Source this file after setting WORK to the allocated project directory.
export CUDA_VISIBLE_DEVICES=
if [[ ${WORK:-} != /* || ${WORK} == / || ! -d ${WORK} ]]; then
    printf '%s\n' 'Set WORK to the existing, allocated project directory.' >&2
    return 1 2>/dev/null || exit 1
fi

export WORK
export GPU_IDS="${GPU_IDS:-6,7}"
if [[ ! $GPU_IDS =~ ^[0-9]+,[0-9]+$ || ${GPU_IDS%,*} == "${GPU_IDS#*,}" ]]; then
    printf '%s\n' 'GPU_IDS must contain two distinct allocated GPU indices.' >&2
    return 1 2>/dev/null || exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export MINEMEM_PROXY="${MINEMEM_PROXY-http://127.0.0.1:17890}"
export http_proxy="${http_proxy:-${HTTP_PROXY:-$MINEMEM_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-$MINEMEM_PROXY}}"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export no_proxy="${no_proxy:-${NO_PROXY:-localhost,127.0.0.1,::1}}"
export NO_PROXY="$no_proxy"

export XDG_CACHE_HOME="$WORK/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TMPDIR="$WORK/tmp"
export MINESTUDIO_DIR="$WORK/runtime/MineStudio"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MINESTUDIO_GPU_RENDER=0
