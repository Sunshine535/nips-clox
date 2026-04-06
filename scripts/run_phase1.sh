#!/bin/bash
# CLOX Phase 1: Qwen2.5-7B on GSM8K + StrategyQA (crossover validation)
# 2× H100 parallel, 200 examples, 5 seeds, 8 strategies
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
PYTHON=/workspace/nips-clox/venv/bin/python
CODE=/workspace/nips-clox/code
RESULTS=/workspace/nips-clox/results/full/qwen2.5-7b-instruct
LOGS=/workspace/nips-clox/logs
mkdir -p "$RESULTS" "$LOGS"

MODEL="Qwen/Qwen2.5-7B-Instruct"
STRATEGIES="standard_cot,self_consistency,backward_cloze,uncertainty_masked_repair,random_masked_repair,full_regeneration,hierarchical_repair,clox_adaptive"
SEEDS="11,23,37,47,59"
MAX_EX=200

echo "=== CLOX Phase 1: Crossover Validation ==="
echo "Model: $MODEL | Examples: $MAX_EX | Seeds: $SEEDS"
echo "Start: $(date)"

# GPU 0: GSM8K (predicted: masking-advantage, low EPL, high recoverability)
CUDA_VISIBLE_DEVICES=0 $PYTHON $CODE/main.py \
    --model "$MODEL" \
    --benchmarks gsm8k \
    --strategies "$STRATEGIES" \
    --seeds "$SEEDS" \
    --max_examples $MAX_EX \
    --max_new_tokens 512 \
    --output_dir "$RESULTS" \
    --n_gpus 1 \
    --log_file "$LOGS/phase1_gsm8k.log" \
    2>&1 | sed 's/^/[GSM8K] /' &
PID0=$!

# GPU 1: StrategyQA (predicted: SC-advantage, high EPL, low recoverability)
CUDA_VISIBLE_DEVICES=1 $PYTHON $CODE/main.py \
    --model "$MODEL" \
    --benchmarks strategyqa \
    --strategies "$STRATEGIES" \
    --seeds "$SEEDS" \
    --max_examples $MAX_EX \
    --max_new_tokens 512 \
    --output_dir "$RESULTS" \
    --n_gpus 1 \
    --log_file "$LOGS/phase1_strategyqa.log" \
    2>&1 | sed 's/^/[SQA ] /' &
PID1=$!

echo "GPU 0 PID: $PID0 (GSM8K)"
echo "GPU 1 PID: $PID1 (StrategyQA)"

wait $PID0
echo "[GSM8K] DONE at $(date)"
wait $PID1
echo "[SQA] DONE at $(date)"

echo "=== Phase 1 COMPLETE at $(date) ==="

# Immediately start Phase 2
echo "=== Starting Phase 2: MATH + ARC + BBH ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON $CODE/main.py \
    --model "$MODEL" \
    --benchmarks math,arc_challenge \
    --strategies "$STRATEGIES" \
    --seeds "$SEEDS" \
    --max_examples $MAX_EX \
    --max_new_tokens 512 \
    --output_dir "$RESULTS" \
    --n_gpus 1 \
    --log_file "$LOGS/phase2_math_arc.log" \
    2>&1 | sed 's/^/[MATH+ARC] /' &
PID2=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON $CODE/main.py \
    --model "$MODEL" \
    --benchmarks bbh \
    --strategies "$STRATEGIES" \
    --seeds "$SEEDS" \
    --max_examples $MAX_EX \
    --max_new_tokens 512 \
    --output_dir "$RESULTS" \
    --n_gpus 1 \
    --log_file "$LOGS/phase2_bbh.log" \
    2>&1 | sed 's/^/[BBH ] /' &
PID3=$!

wait $PID2
echo "[MATH+ARC] DONE at $(date)"
wait $PID3
echo "[BBH] DONE at $(date)"

echo "=== Phase 2 COMPLETE at $(date) ==="

# Phase 3: Cross-model replication with Mistral
echo "=== Starting Phase 3: Mistral cross-model replication ==="
MODEL2="mistralai/Mistral-7B-Instruct-v0.3"
RESULTS2=/workspace/nips-clox/results/full/mistral-7b-instruct-v0.3
mkdir -p "$RESULTS2"

CUDA_VISIBLE_DEVICES=0 $PYTHON $CODE/main.py \
    --model "$MODEL2" \
    --benchmarks gsm8k \
    --strategies "$STRATEGIES" \
    --seeds "11,23,37" \
    --max_examples $MAX_EX \
    --max_new_tokens 512 \
    --output_dir "$RESULTS2" \
    --n_gpus 1 \
    --log_file "$LOGS/phase3_mistral_gsm8k.log" \
    2>&1 | sed 's/^/[MISTRAL-GSM8K] /' &
PID4=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON $CODE/main.py \
    --model "$MODEL2" \
    --benchmarks strategyqa \
    --strategies "$STRATEGIES" \
    --seeds "11,23,37" \
    --max_examples $MAX_EX \
    --max_new_tokens 512 \
    --output_dir "$RESULTS2" \
    --n_gpus 1 \
    --log_file "$LOGS/phase3_mistral_sqa.log" \
    2>&1 | sed 's/^/[MISTRAL-SQA ] /' &
PID5=$!

wait $PID4
echo "[MISTRAL-GSM8K] DONE at $(date)"
wait $PID5
echo "[MISTRAL-SQA] DONE at $(date)"

echo "=== ALL EXPERIMENTS COMPLETE at $(date) ==="
