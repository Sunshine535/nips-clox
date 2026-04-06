#!/bin/bash
# CLOX: Environment setup with robust Python detection
# Handles conda-only servers where python3 is not in PATH
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== CLOX Environment Setup ==="
echo "Project: ${SCRIPT_DIR}"

# ── Find Python >= 3.9 ───────────────────────────────────────────
find_python() {
    local candidates=(
        python3
        python
        /root/miniconda3/bin/python
        /opt/conda/bin/python
        /usr/local/bin/python3
        /usr/bin/python3
    )
    if [ -n "${CONDA_PREFIX:-}" ]; then
        candidates=("${CONDA_PREFIX}/bin/python" "${candidates[@]}")
    fi
    if [ -n "${HOME:-}" ]; then
        candidates+=(
            "${HOME}/miniconda3/bin/python"
            "${HOME}/anaconda3/bin/python"
            "${HOME}/.local/bin/python3"
        )
    fi
    for candidate in "${candidates[@]}"; do
        if [ ! -x "${candidate}" ] 2>/dev/null && ! command -v "${candidate}" &>/dev/null 2>&1; then
            continue
        fi
        if "${candidate}" -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

PYTHON="${CLOX_PYTHON:-}"
if [ -z "${PYTHON}" ]; then
    PYTHON=$(find_python) || {
        echo "ERROR: No Python >= 3.9 found." >&2
        echo "Searched: python3, python, conda paths, /root/miniconda3, /opt/conda, ~/miniconda3, ~/anaconda3" >&2
        echo "Set CLOX_PYTHON=/path/to/python to override." >&2
        exit 1
    }
fi

echo "Python: ${PYTHON}"
${PYTHON} --version

# ── Create venv (skip if conda is active or venv exists) ─────────
VENV_DIR="${SCRIPT_DIR}/.venv"
if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "Conda environment detected (${CONDA_PREFIX}), skipping venv creation."
    echo "Installing into current conda env."
elif [ -d "${VENV_DIR}" ]; then
    echo "Activating existing venv: ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    PYTHON="python"
else
    echo "Creating virtual environment: ${VENV_DIR}"
    if "${PYTHON}" -m venv "${VENV_DIR}" 2>/dev/null; then
        source "${VENV_DIR}/bin/activate"
        PYTHON="python"
        echo "Activated venv: ${VENV_DIR}"
    else
        echo "WARNING: venv creation failed, installing into user/system Python."
    fi
fi

# ── Install dependencies ──────────────────────────────────────────
echo ""
echo "=== Installing dependencies ==="
${PYTHON} -m pip install --upgrade pip -q
${PYTHON} -m pip install -r "${SCRIPT_DIR}/code/requirements.txt" -q

# ── Verify ────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="
${PYTHON} -c "
import torch, transformers, datasets, numpy, scipy
print(f'  torch         {torch.__version__}')
print(f'  transformers  {transformers.__version__}')
print(f'  datasets      {datasets.__version__}')
print(f'  numpy         {numpy.__version__}')
print(f'  scipy         {scipy.__version__}')
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f'  CUDA GPUs:    {n}')
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f'    GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB')
else:
    print('  CUDA: not available')
"

echo ""
echo "=== Setup complete ==="
echo "Run experiments:"
echo "  bash scripts/run_all_experiments.sh"
