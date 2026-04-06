#!/bin/bash
# CLOX v2: Full Experiment Pipeline
# Run on 2×H100 80GB with Qwen3-32B
set -e

VENV="/home/claude/clox-venv"
CODE="/home/claude/nips-clox/code"
RESULTS="/home/claude/nips-clox/results/v2"

source "$VENV/bin/activate"
cd "$CODE"

MODEL="Qwen/Qwen3-32B"
TP=2
MAX_TOKENS=1024
SEEDS="11,23,37,47,59"

echo "============================================"
echo "CLOX v2: Full Experiment Pipeline"
echo "Model: $MODEL (TP=$TP)"
echo "Started: $(date)"
echo "============================================"

# Phase 1: Topology characterization (all benchmarks)
echo ""
echo "=== Phase 1: Topology Characterization ==="
for BENCH in gsm8k math strategyqa arc_challenge; do
    echo "--- $BENCH ---"
    python3 run_clox.py \
        --model "$MODEL" --tp "$TP" \
        --benchmarks "$BENCH" \
        --phase topology \
        --max_tokens "$MAX_TOKENS" \
        --output_dir "$RESULTS" \
        --n_pilot 8 \
        --log_file "$RESULTS/phase1_${BENCH}.log" \
        2>&1 | tee -a "$RESULTS/phase1.log"
done

# Phase 2: Strategy comparison (core strategies)
echo ""
echo "=== Phase 2: Strategy Comparison ==="
STRATEGIES="standard_cot,self_consistency,compute_matched_sc,targeted_repair,random_repair,backward_cloze,full_regeneration,hierarchical_repair"

for BENCH in gsm8k math strategyqa arc_challenge; do
    echo "--- $BENCH ---"
    python3 run_clox.py \
        --model "$MODEL" --tp "$TP" \
        --benchmarks "$BENCH" \
        --phase strategies \
        --strategies "$STRATEGIES" \
        --seeds "$SEEDS" \
        --max_tokens "$MAX_TOKENS" \
        --output_dir "$RESULTS" \
        --log_file "$RESULTS/phase2_${BENCH}.log" \
        2>&1 | tee -a "$RESULTS/phase2.log"
done

# Phase 3: Adaptive routing
echo ""
echo "=== Phase 3: Adaptive Routing ==="
for BENCH in gsm8k math strategyqa arc_challenge; do
    echo "--- $BENCH ---"
    python3 run_clox.py \
        --model "$MODEL" --tp "$TP" \
        --benchmarks "$BENCH" \
        --phase adaptive \
        --seeds "$SEEDS" \
        --max_tokens "$MAX_TOKENS" \
        --output_dir "$RESULTS" \
        --log_file "$RESULTS/phase3_${BENCH}.log" \
        2>&1 | tee -a "$RESULTS/phase3.log"
done

echo ""
echo "============================================"
echo "CLOX v2: Pipeline Complete"
echo "Finished: $(date)"
echo "Results in: $RESULTS"
echo "============================================"
