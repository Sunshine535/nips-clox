#!/bin/bash
# Meta-sweep: 4 models in parallel, each covers 4 benchmarks with 3 strategies
# Wall time: ~1h for smallest, ~3h for 9B
set -e
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/openbayes/input/input0}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BENCHMARKS="gsm8k,math_hard,strategyqa,bbh_logic"
N=30

# (model_name, gpu_pair, tp, gpu_mem) tuples
declare -a CELLS=(
    "Qwen2.5-1.5B|/openbayes/input/input0/hub/Qwen2.5-1.5B|0,1|2|0.50"
    "Qwen2.5-3B|/openbayes/input/input0/hub/Qwen2.5-3B|2,3|2|0.55"
    "Qwen2.5-7B|/openbayes/input/input0/hub/Qwen2.5-7B|4,5|2|0.70"
    "Qwen3.5-9B|/openbayes/input/input0/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a|6,7|2|0.75"
)

mkdir -p results/meta

for entry in "${CELLS[@]}"; do
    IFS='|' read -r TAG MODEL GPUS TP MEM <<< "$entry"
    OUT="results/meta/$TAG"
    mkdir -p "$OUT"

    CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup python3 -u code/meta_sweep.py \
        --model "$MODEL" \
        --benchmarks "$BENCHMARKS" \
        --n_problems $N \
        --tp $TP \
        --gpu_mem $MEM \
        --output "$OUT" \
        </dev/null >"$OUT/run.log" 2>&1 &
    disown
    echo "  $TAG (GPUs $GPUS, PID $!) -> $OUT/"
done

echo ""
echo "Monitor: tail -f results/meta/*/sweep.log"
echo "Analyze after: python3 code/analyze_meta.py results/meta"
