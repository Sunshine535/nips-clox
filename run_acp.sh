#!/bin/bash
# SenseCore cluster launcher for CLOX experiments
# Wraps scripts/run_all_experiments.sh with cluster-specific paths
set -euo pipefail

PROJECT_DIR=/data/szs/250010072/nwh/nips-clox
DATA_DIR=/data/szs/share/clox
SHARE_DIR=/data/szs/share
mkdir -p "${DATA_DIR}"/{results,logs,hf_cache}

# ── Environment ──────────────────────────────────────────────────
export HF_HOME="${DATA_DIR}/hf_cache"
export HF_DATASETS_CACHE="${DATA_DIR}/hf_cache/datasets"
export TRANSFORMERS_CACHE="${DATA_DIR}/hf_cache/hub"
export TOKENIZERS_PARALLELISM=false
export CLOX_LOG_DIR="${DATA_DIR}/logs"
export CLOX_RESULTS_DIR="${DATA_DIR}/results"

# ── Symlinks ─────────────────────────────────────────────────────
ln -sfn "${DATA_DIR}/results" "${PROJECT_DIR}/results"
ln -sfn "${DATA_DIR}/logs"    "${PROJECT_DIR}/logs"

# ── GPU diagnostics ──────────────────────────────────────────────
echo "=== GPU Diagnostics ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# ── Find Python (handle conda-only servers) ──────────────────────
PYTHON=""
for candidate in python3 python /root/miniconda3/bin/python "${CONDA_PREFIX:-__skip__}/bin/python"; do
    if [ "${candidate}" = "__skip__/bin/python" ]; then continue; fi
    if command -v "${candidate}" &>/dev/null 2>&1; then
        if "${candidate}" -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null; then
            PYTHON="${candidate}"
            break
        fi
    fi
done
if [ -z "${PYTHON}" ]; then
    echo "ERROR: No Python >= 3.9 found" >&2
    exit 1
fi
echo "Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"

${PYTHON} -c "
import torch
n = torch.cuda.device_count()
print(f'PyTorch sees {n} GPUs')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB')
"

# ── Install + Run ────────────────────────────────────────────────
cd "${PROJECT_DIR}"
${PYTHON} -m pip install -q -r code/requirements.txt

exec bash scripts/run_all_experiments.sh
