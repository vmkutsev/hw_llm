#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
run_id=$1
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
run_dir="output_dir/experiments/$run_id"
mkdir -p "$run_dir"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
    --format=csv -l 5 > "$run_dir/gpu.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT
sudo docker image inspect hse-llm:baseline --format '{{.Id}}' > "$run_dir/image_id.txt"
cp output_dir/dataset_manifest.json "$run_dir/dataset_manifest.json"
LLM_RUN_ID="$run_id" bash run.sh python run_experiment.py "configs/$run_id.json" > "$run_dir/train_stdout.log" 2>&1
LLM_RUN_ID="$run_id" bash run.sh python generate.py prompts.json --model "$run_dir" > "$run_dir/generation_greedy.log" 2>&1
LLM_RUN_ID="$run_id" bash run.sh python generate.py prompts.json --model "$run_dir" --sample --output "$run_dir/generations_sample.json" > "$run_dir/generation_sample.log" 2>&1
python3 summarize_run.py "$run_dir"
python3 prune_checkpoints.py "$run_dir"
