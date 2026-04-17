#!/bin/bash
# Download script for CLOX pilot experiment
#
# Purpose: Pre-download all datasets + model on an internet-enabled container,
#          then scp the cache directory to the air-gapped GPU server.
#
# Usage (on download container with internet, China-mainland proxies):
#   cd /openbayes/input/input0
#   git clone https://ghfast.top/https://github.com/Sunshine535/nips-clox.git
#   cd nips-clox
#   bash download_data.sh                    # downloads to /openbayes/input/input0/hf_cache
#   bash download_data.sh /custom/path       # or to custom directory
#
# Usage (on tju-hpc after scp):
#   export HF_HOME=/path/to/scp'd/hf_cache
#   export HF_HUB_OFFLINE=1
#   bash run_pilot.sh --clean --tp 2

set -e

# ── Configuration ───────────────────────────────────────────────────
TARGET_DIR="${1:-/openbayes/input/input0}"
HF_CACHE="$TARGET_DIR/hf_cache"

# Use HF mirror (faster in China; remove if outside China)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1  # faster downloads

MODEL="Qwen/Qwen3.5-27B"

echo "============================================"
echo "CLOX Data Download"
echo "============================================"
echo "Target:      $TARGET_DIR"
echo "HF_HOME:     $HF_HOME"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "Model:       $MODEL"
echo ""

mkdir -p "$HF_CACHE"

# ── Install dependencies (tsinghua mirror) ──────────────────────────
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
echo "[1/3] Installing huggingface_hub, datasets, hf_transfer (via tsinghua)..."
pip install -q -i "$PIP_MIRROR" --trusted-host pypi.tuna.tsinghua.edu.cn \
    huggingface_hub datasets hf_transfer 2>&1 | tail -5

# ── Download datasets ───────────────────────────────────────────────
echo ""
echo "[2/3] Downloading datasets..."
python3 <<'PYEOF'
import os, sys
from datasets import load_dataset

DATASETS = [
    ("gsm8k", "main", "test"),
    ("gsm8k", "main", "train"),
    ("EleutherAI/hendrycks_math", "algebra", "test"),
    ("EleutherAI/hendrycks_math", "counting_and_probability", "test"),
    ("EleutherAI/hendrycks_math", "geometry", "test"),
    ("EleutherAI/hendrycks_math", "intermediate_algebra", "test"),
    ("EleutherAI/hendrycks_math", "number_theory", "test"),
    ("EleutherAI/hendrycks_math", "prealgebra", "test"),
    ("EleutherAI/hendrycks_math", "precalculus", "test"),
    ("ChilleD/StrategyQA", None, "test"),
    ("allenai/ai2_arc", "ARC-Challenge", "test"),
]
BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "logical_deduction_five_objects", "multistep_arithmetic_two",
    "navigate", "tracking_shuffled_objects_three_objects",
]
for s in BBH_SUBTASKS:
    DATASETS.append(("lukaemon/bbh", s, "test"))

ok, fail = 0, 0
for name, config, split in DATASETS:
    tag = f"{name}" + (f"/{config}" if config else "") + f":{split}"
    try:
        if config:
            load_dataset(name, config, split=split)
        else:
            load_dataset(name, split=split)
        print(f"  [OK]   {tag}")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] {tag}: {str(e)[:100]}")
        fail += 1

print(f"\nDatasets: {ok} OK, {fail} FAIL")
if fail > 0:
    sys.exit(1)
PYEOF

# ── Download model ──────────────────────────────────────────────────
echo ""
echo "[3/3] Downloading model $MODEL (~54GB, this takes a while)..."
python3 <<PYEOF
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="$MODEL",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.py"],
)
print(f"Model cached at: {path}")
PYEOF

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Download Complete"
echo "============================================"
du -sh "$HF_CACHE"
echo ""
echo "Next steps:"
echo "  1. scp to tju-hpc:"
echo "     scp -r $TARGET_DIR tju-hpc:~/input0"
echo ""
echo "  2. On tju-hpc, set env vars (offline mode, uses local cache):"
echo "     export HF_HOME=\$HOME/input0/hf_cache"
echo "     export HF_HUB_OFFLINE=1"
echo "     export TRANSFORMERS_OFFLINE=1"
echo ""
echo "  3. Run pilot:"
echo "     cd ~/input0/nips-clox"
echo "     bash run_pilot.sh --clean --tp 2"
