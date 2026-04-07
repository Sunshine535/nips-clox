# CLOX v2 — Topology-Aware Compute-Optimal Inference

Reasoning traces have measurable structural properties — **local recoverability (r̄)** and **error propagation length (ℓ)** — that predict which inference-time strategy is optimal. CLOX estimates these from cheap pilot traces and routes to the best strategy, matching Self-Consistency accuracy at 40-60% compute cost.

## Current State

| Component | Status |
|-----------|--------|
| Synthetic DAG validation | ✅ 83% theory prediction accuracy |
| Topology characterization (32B, 4 benchmarks × 200) | ✅ Complete |
| Topology characterization (8B, 3 benchmarks × 200) | ✅ Complete |
| Pilot results (32B + 8B) | ✅ Complete |
| **Strategy comparison (9 strategies × 3 seeds × 4 benchmarks)** | ❌ **Not started** |
| CLOX-Adaptive evaluation | ❌ Not started |
| Proxy validation | ❌ Not started |
| Paper draft | 🔄 Needs results update |
| Review score | 6.5/10 (Round 3) |

## Resume: Run Remaining Experiments

### Prerequisites

```bash
cd /workspace/nips-clox
source venv/bin/activate

# Verify GPU setup (auto-detects GPU count and sets TP)
python3 code/verify_gpu.py
```

### Step 1: Strategy Comparison (32B)

The main missing piece. 9 strategies × 3 seeds × 4 benchmarks on Qwen2.5-32B-Instruct-AWQ.

GPU auto-detection is built in — `--tp 0` (default) picks TP from model size and available GPUs.

```bash
# Full run: all 4 benchmarks, all 9 strategies, 3 seeds
# TP auto-detected (32B → TP=2 if ≥2 GPUs)
# Checkpoint: saves every 50 examples, safe to kill and re-run
python3 code/run_full_experiment.py \
    --phase strategies \
    --seeds 11,23,37 \
    --output results/v5 \
    --log_file results/v5/strategies.log
```

Or run benchmarks incrementally:

```bash
# GSM8K first (most informative, ~2h)
python3 code/run_full_experiment.py \
    --phase strategies --benchmarks gsm8k \
    --seeds 11,23,37 --output results/v5

# Then MATH (~3h)
python3 code/run_full_experiment.py \
    --phase strategies --benchmarks math \
    --seeds 11,23,37 --output results/v5

# Then StrategyQA + ARC
python3 code/run_full_experiment.py \
    --phase strategies --benchmarks strategyqa,arc_challenge \
    --seeds 11,23,37 --output results/v5
```

Strategies run: `standard_cot`, `self_consistency` (k=5), `compute_matched_sc` (k=2), `targeted_repair`, `random_repair`, `backward_cloze`, `full_regeneration`, `hierarchical_repair`, `clox_adaptive`.

### Step 2: Proxy Validation

Tests how many pilot traces are needed for reliable topology estimation.

```bash
python3 code/run_full_experiment.py \
    --phase proxy \
    --output results/v5
```

### Step 3: Analysis & Figures

```bash
python3 code/analyze_v2.py results/v5/Qwen2.5-32B-Instruct-AWQ/
```

### Step 4: Cross-Model (Optional)

Run the same on Qwen3-8B for cross-model analysis:

```bash
python3 code/run_full_experiment.py \
    --model Qwen/Qwen3-8B \
    --phase strategies \
    --seeds 11,23,37 \
    --output results/v5
```

## Checkpoint & Resume

`run_full_experiment.py` checkpoints every 50 examples per (strategy, seed) combo:

- `.ckpt_{strategy}_s{seed}.json` files in each benchmark directory
- Re-running the same command skips completed combos automatically
- To force re-run a specific combo: delete its `.ckpt_*.json` file
- Final results are saved to `{benchmark}/strategies.json` with aggregate statistics

## GPU Auto-Detection

The engine auto-detects available GPUs and picks tensor parallelism:

| Model size | TP (4 GPUs) | TP (2 GPUs) | TP (1 GPU) |
|-----------|-------------|-------------|------------|
| 70B/72B | 4 | 2 | 1 |
| 32B/34B | 2 | 2 | 1 |
| ≤14B | 1 | 1 | 1 |

Override with `--tp N`. Respects `CUDA_VISIBLE_DEVICES`.

## Existing Results

| Data | Location | Notes |
|------|----------|-------|
| Synthetic DAG | `results/synthetic/` | 5 graph types × 6 r̄ × 3 seeds × 2000 trials |
| 32B Topology | `results/v3/Qwen2.5-32B-Instruct-AWQ/` | r̄, ℓ for GSM8K/MATH/StrategyQA/ARC |
| 8B Topology | `results/v4/Qwen3-8B/` | r̄, ℓ for MATH/StrategyQA/ARC |
| 32B Pilot | `results/v3/.../pilot/pilot_results.json` | 50 examples × 5 strategies × 4 benchmarks |
| 8B Pilot | `results/v4/.../pilot/pilot_results.json` | 50 examples × 5 strategies × 4 benchmarks |

### Key Topology Numbers (32B)

| Benchmark | r̄ | ℓ | Predicted Strategy |
|-----------|-----|------|-------------------|
| GSM8K | 0.521 ± 0.074 | 1.28 ± 0.28 | Targeted repair |
| MATH | 0.634 ± 0.086 | 1.18 ± 0.28 | Targeted repair |
| StrategyQA | 0.451 ± 0.057 | 1.55 ± 0.28 | Standard CoT / Adaptive |
| ARC-Challenge | 0.427 ± 0.055 | 1.57 ± 0.38 | Standard CoT / Adaptive |

## Project Structure

```
code/
  engine.py              # vLLM engine (auto GPU detection + TP)
  strategies_v2.py       # 9 inference strategies
  topology_v2.py         # Topology estimation (r̄, ℓ)
  run_full_experiment.py # Main runner (checkpoint + auto-TP)
  run_clox.py            # Phase-based runner
  synthetic_dag.py       # Synthetic DAG validation
  benchmarks.py          # Benchmark loaders
  evaluation.py          # Statistical tests
  analyze_v2.py          # Analysis + figures
  verify_gpu.py          # GPU verification script
results/
  synthetic/             # DAG validation
  v3/                    # 32B results (topology + pilot)
  v4/                    # 8B results (topology + pilot)
  v5/                    # [target] Full strategy comparison
paper/
  main.tex               # NeurIPS draft
```
