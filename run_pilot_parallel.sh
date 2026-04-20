#!/bin/bash
# Data-parallel pilot launcher: 4 shards × 2 GPUs (tp=2)
# Each shard runs its own vLLM instance on its own GPU pair and processes
# a subset of the 50 problems. Merge with: python3 code/merge_shards.py results/pilot
set -e
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/openbayes/input/input0}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-/openbayes/input/input0/hub/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd}"
N_SHARDS=4
TP=2
GPU_MEM=0.85

mkdir -p results/pilot

echo "Launching $N_SHARDS shards, TP=$TP each (8 GPUs total)"
echo "Model: $MODEL"

for i in $(seq 0 $((N_SHARDS-1))); do
    GPU_LO=$((i * TP))
    GPU_HI=$((GPU_LO + TP - 1))
    GPUS=$(seq -s, $GPU_LO $GPU_HI)
    OUT="results/pilot/shard_$i"
    mkdir -p "$OUT"

    CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup python3 -u code/run_pilot.py \
        --model "$MODEL" \
        --tp $TP \
        --gpu_mem $GPU_MEM \
        --shard_id $i \
        --n_shards $N_SHARDS \
        --output "$OUT" \
        --clean \
        </dev/null >"$OUT/pilot_run.log" 2>&1 &
    disown
    PID=$!
    echo "  Shard $i (GPUs $GPUS, PID $PID) -> $OUT/"
done

echo ""
echo "Monitor: tail -f results/pilot/shard_*/pilot.log"
echo "When done: python3 code/merge_shards.py results/pilot"
