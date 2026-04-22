#!/bin/bash
# Validate top-3 cells at N=150 (vs pilot N=30) to tighten CI on Oracle-SC gap
set -e
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/openbayes/input/input0}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

N=150

# (tag, model_path, gpu_pair, bench_str, gpu_mem)
declare -a CELLS=(
    "1.5B_strategyqa|/openbayes/input/input0/hub/Qwen2.5-1.5B|0,1|strategyqa|0.50"
    "7B_strategyqa|/openbayes/input/input0/hub/Qwen2.5-7B|2,3|strategyqa|0.70"
    "3B_bbh_logic|/openbayes/input/input0/hub/Qwen2.5-3B|4,5|bbh_logic|0.55"
)

mkdir -p results/meta_n150

for entry in "${CELLS[@]}"; do
    IFS='|' read -r TAG MODEL GPUS BENCH MEM <<< "$entry"
    OUT="results/meta_n150/$TAG"
    mkdir -p "$OUT"

    CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup python3 -u code/meta_sweep.py \
        --model "$MODEL" \
        --benchmarks "$BENCH" \
        --n_problems $N \
        --tp 2 \
        --gpu_mem $MEM \
        --output "$OUT" \
        </dev/null >"$OUT/run.log" 2>&1 &
    disown
    echo "  $TAG (GPUs $GPUS, PID $!) -> $OUT/"
done

echo ""
echo "Monitor: tail -f results/meta_n150/*/sweep.log"
