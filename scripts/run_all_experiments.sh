#!/bin/bash
# CLOX: Full experiment pipeline
# Runs smoke test → full 3-model × 5-benchmark × 8-strategy × 5-seed study
# Supports skip-if-done, proper logging, multi-GPU auto-detection
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

# ── Configuration (override via env) ──────────────────────────────
BENCHMARKS="${CLOX_BENCHMARKS:-gsm8k,math,strategyqa,arc_challenge,bbh}"
STRATEGIES="${CLOX_STRATEGIES:-all}"
SEEDS="${CLOX_SEEDS:-11,23,37,47,59}"
MAX_NEW_TOKENS="${CLOX_MAX_TOKENS:-512}"
LOG_DIR="${CLOX_LOG_DIR:-${PROJECT_DIR}/logs}"
RESULTS_DIR="${CLOX_RESULTS_DIR:-${PROJECT_DIR}/results}"
QUANTIZE="${CLOX_QUANTIZE:-}"

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

# ── Detect Python ─────────────────────────────────────────────────
find_python() {
    local candidates=(
        python3
        python
        /root/miniconda3/bin/python
        /opt/conda/bin/python
        "${HOME}/miniconda3/bin/python"
        "${HOME}/anaconda3/bin/python"
    )
    if [ -n "${CONDA_PREFIX:-}" ]; then
        candidates=("${CONDA_PREFIX}/bin/python" "${candidates[@]}")
    fi
    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" &>/dev/null 2>&1; then
            if "${candidate}" -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null; then
                echo "${candidate}"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=$(find_python) || {
    echo "ERROR: No Python >= 3.9 found." >&2
    exit 1
}
echo "Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"

# ── Detect GPUs ───────────────────────────────────────────────────
N_GPUS=1
if command -v nvidia-smi &>/dev/null; then
    N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    if [ "${N_GPUS}" -lt 1 ]; then N_GPUS=1; fi
    echo "GPUs detected: ${N_GPUS}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
else
    echo "No nvidia-smi found, using CPU mode (n_gpus=1)"
fi

# ── Environment ───────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# ── Install dependencies ──────────────────────────────────────────
echo "=== Installing dependencies ==="
${PYTHON} -m pip install -q -r code/requirements.txt

# ── Helper: check if experiment is complete ───────────────────────
is_done() {
    local output_dir="$1"
    local summary="${output_dir}/experiment_summary.json"
    [ -f "${summary}" ]
}

# ── Quantize flag ─────────────────────────────────────────────────
QUANTIZE_FLAG=""
if [ -n "${QUANTIZE}" ]; then
    QUANTIZE_FLAG="--quantize"
fi

# ── Phase 1: Smoke test ──────────────────────────────────────────
SMOKE_DIR="${RESULTS_DIR}/smoke"
if is_done "${SMOKE_DIR}"; then
    echo "[SKIP] Smoke test already complete: ${SMOKE_DIR}/experiment_summary.json"
else
    echo ""
    echo "=========================================="
    echo "  Phase 1: Smoke Test"
    echo "=========================================="
    SMOKE_LOG="${LOG_DIR}/smoke_$(date +%Y%m%d_%H%M%S).log"
    ${PYTHON} code/main.py \
        --model "${MODELS[0]}" \
        --benchmarks gsm8k \
        --strategies standard_cot,self_consistency \
        --seeds 11 \
        --max_examples 5 \
        --max_new_tokens 256 \
        --output_dir "${SMOKE_DIR}" \
        --log_file "${SMOKE_LOG}" \
        ${QUANTIZE_FLAG} \
        2>&1 | tee -a "${SMOKE_LOG}"

    if is_done "${SMOKE_DIR}"; then
        echo "[OK] Smoke test passed"
    else
        echo "[FAIL] Smoke test did not produce experiment_summary.json" >&2
        exit 1
    fi
fi

# ── Phase 2: Full experiments per model ───────────────────────────
echo ""
echo "=========================================="
echo "  Phase 2: Full Experiments"
echo "  Models:     ${#MODELS[@]}"
echo "  Benchmarks: ${BENCHMARKS}"
echo "  Strategies: ${STRATEGIES}"
echo "  Seeds:      ${SEEDS}"
echo "  GPUs:       ${N_GPUS}"
echo "=========================================="

TOTAL_MODELS=${#MODELS[@]}
MODEL_IDX=0
FAILED=0

for MODEL in "${MODELS[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))
    MODEL_TAG=$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '.' '_')
    OUTPUT_DIR="${RESULTS_DIR}/${MODEL_TAG}"

    if is_done "${OUTPUT_DIR}"; then
        echo "[SKIP] (${MODEL_IDX}/${TOTAL_MODELS}) ${MODEL_TAG}: already complete"
        continue
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_LOG="${LOG_DIR}/clox_${MODEL_TAG}_${TIMESTAMP}.log"

    echo ""
    echo "── (${MODEL_IDX}/${TOTAL_MODELS}) ${MODEL_TAG} ──"
    echo "  Model:   ${MODEL}"
    echo "  Output:  ${OUTPUT_DIR}"
    echo "  Log:     ${RUN_LOG}"

    if ${PYTHON} code/main.py \
        --model "${MODEL}" \
        --benchmarks "${BENCHMARKS}" \
        --strategies "${STRATEGIES}" \
        --seeds "${SEEDS}" \
        --n_gpus "${N_GPUS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${OUTPUT_DIR}" \
        --log_file "${RUN_LOG}" \
        ${QUANTIZE_FLAG} \
        2>&1 | tee -a "${RUN_LOG}"; then
        echo "[OK] ${MODEL_TAG} complete"
    else
        echo "[WARN] ${MODEL_TAG} exited with error (see ${RUN_LOG})" >&2
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Pipeline Complete"
echo "=========================================="
echo "  Results: ${RESULTS_DIR}/"
echo "  Logs:    ${LOG_DIR}/"

DONE_COUNT=0
for MODEL in "${MODELS[@]}"; do
    MODEL_TAG=$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '.' '_')
    if is_done "${RESULTS_DIR}/${MODEL_TAG}"; then
        echo "  [DONE] ${MODEL_TAG}"
        DONE_COUNT=$((DONE_COUNT + 1))
    else
        echo "  [MISS] ${MODEL_TAG}"
    fi
done

echo ""
echo "Completed: ${DONE_COUNT}/${TOTAL_MODELS} models"
if [ "${FAILED}" -gt 0 ]; then
    echo "WARNING: ${FAILED} model(s) had errors. Check logs."
    exit 1
fi
