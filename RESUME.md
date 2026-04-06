# CLOX v2: Resume Guide

## Quick Start

```bash
# 1. Create venv and install deps
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install vllm transformers datasets numpy scipy matplotlib sentencepiece

# 2. Run experiment (adjust GPU and model as needed)
cd code
# With vLLM (if GPUs are clean, ~160GB available):
python3 run_clox.py --model Qwen/Qwen3-32B --tp 2 --benchmarks gsm8k,strategyqa,arc_challenge --phase all

# With HuggingFace (if GPU memory constrained):
CUDA_VISIBLE_DEVICES=0 python3 run_focused.py

# 3. Analyze results
python3 analyze_v2.py results/v2_focused/
```

## Current Results (2026-04-05)

### GSM8K (200 examples, Qwen2.5-7B-Instruct)
| Strategy | Accuracy | Tokens | Cost |
|---|---:|---:|---:|
| Standard CoT | 78.5% | 598 | 1.0× |
| SC-5 (k=5) | **87.0%** | 3043 | 5.1× |
| CM-SC (k=2) | 68.5% | 1211 | 2.0× |
| Targeted Repair | 70.5% | 1352 | 2.3× |
| Random Repair | 70.5% | 1314 | 2.2× |
| **Backward Cloze** | **86.0%** | 2133 | 3.6× |
| Full Regen | 55.5% | 1306 | 2.2× |

### StrategyQA (200 examples) — 7B too weak (~50% all strategies)
### ARC-Challenge (200 examples) — CoT=89.5%, SC=90%

### Topology (Qwen3-32B-AWQ, GSM8K)
r̄=0.456±0.059, ℓ=2.86±2.55

## What's Needed for NeurIPS Submission

1. **GPU cleanup** — reboot to clear zombie CUDA contexts, then use 2×H100 with TP=2
2. **Larger model** — Qwen3-32B or Qwen2.5-72B-Instruct-AWQ (7B too weak for StrategyQA)
3. **Full-scale experiments** — 1319 GSM8K × 5 seeds × all strategies
4. **Multi-benchmark topology** — characterize StrategyQA, ARC, MATH topology
5. **CLOX-Adaptive** — run adaptive routing evaluation
6. **Paper writing** — update results tables in paper/main.tex

## Key Files

- `code/engine.py` — vLLM generation engine
- `code/strategies_v2.py` — 9 strategies (use these, not strategies.py)
- `code/topology_v2.py` — topology estimation
- `code/run_clox.py` — main 3-phase runner (vLLM)
- `code/run_focused.py` — focused experiment (HF generate fallback)
- `code/run_32b.py` — Qwen3-32B specific runner
- `code/analyze_v2.py` — analysis + figures
- `results/v2_focused/` — all current results
- `results/v2_32b/gsm8k/topology.json` — topology data
- `paper/main.tex` — NeurIPS paper (744 lines, needs results update)
