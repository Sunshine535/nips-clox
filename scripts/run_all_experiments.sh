#!/bin/bash
# Phase 1: Fill missing cells for Qwen3-8B (all 5) + Qwen3.5-9B/ARC + Qwen3.5-27B (4 missing)
# All 8 GPUs used simultaneously
set -e
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/openbayes/input/input0}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

QWEN3_8B="/openbayes/input/input0/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
QWEN35_9B="/openbayes/input/input0/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN35_27B="/openbayes/input/input0/hub/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd"

N=30

echo "=== PHASE 1: Fill missing cells (3 models in parallel) ==="
mkdir -p results/meta/Qwen3-8B results/meta/Qwen3.5-9B results/meta/Qwen3.5-27B

# Qwen3-8B: all 5 benchmarks (tp=2, GPU 0,1)
CUDA_VISIBLE_DEVICES=0,1 setsid nohup python3 -u code/meta_sweep.py \
    --model "$QWEN3_8B" \
    --benchmarks gsm8k,math_hard,strategyqa,arc_challenge,bbh_logic \
    --n_problems $N --tp 2 --gpu_mem 0.75 \
    --output results/meta/Qwen3-8B \
    </dev/null >results/meta/Qwen3-8B/run.log 2>&1 &
disown; echo "  Qwen3-8B × 5 bench (GPU 0,1, PID $!)"

# Qwen3.5-9B: ARC only (tp=2, GPU 2,3)
CUDA_VISIBLE_DEVICES=2,3 setsid nohup python3 -u code/meta_sweep.py \
    --model "$QWEN35_9B" \
    --benchmarks arc_challenge \
    --n_problems $N --tp 2 --gpu_mem 0.75 \
    --output results/meta/Qwen3.5-9B \
    </dev/null >results/meta/Qwen3.5-9B/run_arc.log 2>&1 &
disown; echo "  Qwen3.5-9B × ARC (GPU 2,3, PID $!)"

# Qwen3.5-27B: 4 missing benchmarks (tp=4, GPU 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 setsid nohup python3 -u code/meta_sweep.py \
    --model "$QWEN35_27B" \
    --benchmarks math_hard,strategyqa,arc_challenge,bbh_logic \
    --n_problems $N --tp 4 --gpu_mem 0.80 \
    --output results/meta/Qwen3.5-27B \
    </dev/null >results/meta/Qwen3.5-27B/run.log 2>&1 &
disown; echo "  Qwen3.5-27B × 4 bench (GPU 4-7, PID $!)"

echo ""
echo "Monitor: tail -f results/meta/Qwen3-8B/sweep.log results/meta/Qwen3.5-27B/sweep.log"
echo "Check: ps aux | grep meta_sweep | grep -v grep | wc -l"
