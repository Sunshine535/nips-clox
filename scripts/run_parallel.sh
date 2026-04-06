#!/bin/bash
# CLOX Parallel Experiment Runner — 2× H100 80GB
# Two independent GPU-pinned processes, each loads model once
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
PYTHON=/workspace/nips-clox/venv/bin/python
CODE=/workspace/nips-clox/code
RESULTS=/workspace/nips-clox/results/full
LOGS=/workspace/nips-clox/logs
mkdir -p "$RESULTS" "$LOGS"

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
MAX_EX="${2:-300}"
SEEDS="${3:-11,23,37,47,59}"
STRATEGIES="standard_cot,self_consistency,backward_cloze,uncertainty_masked_repair,random_masked_repair,full_regeneration,hierarchical_repair,clox_adaptive"
OUT="$RESULTS/$MODEL_SHORT"
mkdir -p "$OUT"

echo "=== CLOX Parallel Experiments ==="
echo "Model: $MODEL"
echo "Max examples per benchmark: $MAX_EX"
echo "Seeds: $SEEDS"
echo "Output: $OUT"
echo "Start: $(date)"

# GPU 0: GSM8K, MATH, ARC-Challenge (arithmetic/science — predicted masking-friendly)
CUDA_VISIBLE_DEVICES=0 $PYTHON $CODE/main.py \
    --model "$MODEL" \
    --benchmarks gsm8k,math,arc_challenge \
    --strategies "$STRATEGIES" \
    --seeds "$SEEDS" \
    --max_examples "$MAX_EX" \
    --max_new_tokens 512 \
    --output_dir "$OUT" \
    --n_gpus 1 \
    --log_file "$LOGS/${MODEL_SHORT}_gpu0_$(date +%Y%m%d_%H%M%S).log" \
    2>&1 | sed 's/^/[GPU0] /' &
PID0=$!

# GPU 1: StrategyQA, BBH (multi-hop/diverse — predicted SC-friendly)
CUDA_VISIBLE_DEVICES=1 $PYTHON $CODE/main.py \
    --model "$MODEL" \
    --benchmarks strategyqa,bbh \
    --strategies "$STRATEGIES" \
    --seeds "$SEEDS" \
    --max_examples "$MAX_EX" \
    --max_new_tokens 512 \
    --output_dir "$OUT" \
    --n_gpus 1 \
    --log_file "$LOGS/${MODEL_SHORT}_gpu1_$(date +%Y%m%d_%H%M%S).log" \
    2>&1 | sed 's/^/[GPU1] /' &
PID1=$!

echo "GPU 0 PID: $PID0 (gsm8k, math, arc_challenge)"
echo "GPU 1 PID: $PID1 (strategyqa, bbh)"
echo "Waiting for both to finish..."

wait $PID0
STATUS0=$?
echo "GPU 0 finished with status $STATUS0 at $(date)"

wait $PID1
STATUS1=$?
echo "GPU 1 finished with status $STATUS1 at $(date)"

echo "=== ALL COMPLETE at $(date) ==="
exit $((STATUS0 + STATUS1))
