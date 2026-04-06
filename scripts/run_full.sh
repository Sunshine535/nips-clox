#!/bin/bash
# CLOX Full Experiment Suite — 2× H100 80GB
# 3 models × 5 benchmarks × 8 strategies × 5 seeds
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
PYTHON=/workspace/nips-clox/venv/bin/python
CODE=/workspace/nips-clox/code
RESULTS=/workspace/nips-clox/results/full
LOGS=/workspace/nips-clox/logs
mkdir -p "$RESULTS" "$LOGS"

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

BENCHMARKS="gsm8k,math,strategyqa,arc_challenge,bbh"
STRATEGIES="standard_cot,self_consistency,backward_cloze,uncertainty_masked_repair,random_masked_repair,full_regeneration,hierarchical_repair,clox_adaptive"
SEEDS="11,23,37,47,59"
MAX_TOKENS=512

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    LOG="$LOGS/${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S).log"
    OUT="$RESULTS/$MODEL_SHORT"
    mkdir -p "$OUT"

    echo "=== Starting $MODEL_SHORT at $(date) ===" | tee -a "$LOG"

    cd "$CODE"
    $PYTHON main.py \
        --model "$MODEL" \
        --benchmarks "$BENCHMARKS" \
        --strategies "$STRATEGIES" \
        --seeds "$SEEDS" \
        --max_new_tokens $MAX_TOKENS \
        --output_dir "$OUT" \
        --n_gpus 2 \
        --log_file "$LOG" \
        2>&1 | tee -a "$LOG"

    echo "=== Finished $MODEL_SHORT at $(date) ===" | tee -a "$LOG"
done

echo "=== ALL EXPERIMENTS COMPLETE ==="
