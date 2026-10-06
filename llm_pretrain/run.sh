#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .cache
run_id=${LLM_RUN_ID:-baseline}
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
sudo docker run --rm --gpus all --ipc=host \
    --user "$(id -u):$(id -g)" \
    -v "$PWD:/work" -w /work \
    -e HF_HOME=/work/.cache/huggingface \
    -e TORCH_HOME=/work/.cache/torch \
    -e TORCHINDUCTOR_CACHE_DIR="/work/.cache/inductor-$run_id" \
    -e TRITON_CACHE_DIR="/work/.cache/triton-$run_id" \
    -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    -e CUDA_CACHE_PATH=/work/.cache/cuda \
    -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false \
    hse-llm:baseline "$@"
