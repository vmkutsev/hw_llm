#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
run_dir=output_dir/gpt2-1b-russian
mkdir -p "$run_dir"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
    --format=csv -l 5 > "$run_dir/gpu.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT
sudo docker image inspect hse-llm:baseline --format '{{.Id}}' > "$run_dir/image_id.txt"
bash run.sh python your_solution.py train
