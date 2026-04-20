#!/bin/bash
set -e
cd "$(dirname "$0")"

# --- HF cache: 统一落盘到 /openbayes/input/input0 ---
export HF_HOME="${HF_HOME:-/openbayes/input/input0}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" results/pilot

nohup python3 -u code/run_pilot.py "$@" > results/pilot/pilot_run.log 2>&1 &
echo "PID: $!"
echo "Log: tail -f results/pilot/pilot_run.log"
