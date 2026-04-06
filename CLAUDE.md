## Project

CLOX v2: Topology-Aware Compute-Optimal Inference

## Status

Running. Phase 1 topology characterization complete for GSM8K. Phase 2 strategy comparison in progress.

## Key Context

- **Core question:** Can reasoning trace topology (r̄, ℓ) predict which inference strategy is optimal?
- **v1 answer:** Negative on synthetic benchmark with 1.5B model (known bugs, underpowered)
- **v2 approach:** Fixed implementations, real benchmarks, Qwen3-32B-AWQ, proper topology estimation

## Hardware

- 2× NVIDIA H100 80GB HBM3 (using GPU 1 due to orphaned root processes on GPU 0)
- Model: Qwen3-32B-AWQ (4-bit quantized, ~18GB VRAM)
- vLLM 0.19.0 with tensor parallelism

## v2 Changes from v1

1. **Fixed backward cloze** — now actually generates backward from answer to premises
2. **Fixed entropy** — real per-token entropy from top-20 softmax, not fake single-token proxy
3. **Fixed ablation collapse** — random vs targeted masking produce distinct outputs
4. **Real benchmarks** — GSM8K, MATH, StrategyQA, ARC-Challenge
5. **Proper model** — Qwen3-32B (latest) instead of 1.5B
6. **vLLM** — batched inference, 10-50x faster

## First Results (GSM8K Topology, Qwen3-32B-AWQ)

| Difficulty | r̄ (mean±std) | ℓ (mean±std) | N |
|---|---|---|---|
| Easy | 0.467±0.056 | 2.34±2.00 | 116 |
| Medium | 0.444±0.060 | 3.62±3.02 | 73 |
| Hard | 0.420±0.048 | 3.30±2.99 | 11 |
| Overall | 0.456±0.059 | 2.86±2.55 | 200 |

Pattern: Harder problems have lower recoverability and longer error propagation.

## File Layout

```
code/
  engine.py           — vLLM generation engine wrapper
  strategies_v2.py    — Fixed strategy implementations (9 strategies)
  topology_v2.py      — Proper topology estimation (r̄, ℓ)
  run_clox.py         — Main experiment runner (3-phase)
  run_32b.py          — Qwen3-32B specific runner
  benchmarks.py       — Benchmark loaders (unchanged)
  evaluation.py       — Statistical tests (unchanged)
  analyze_v2.py       — Analysis + figure generation

results/
  v2_32b/             — Qwen3-32B-AWQ results (in progress)
  v2_7b/              — Qwen2.5-7B validation results

paper/
  main.tex            — NeurIPS paper draft (theory + experiments)
```

## Running Experiment

PID running via nohup on GPU 1. Log at results/run_32b.log.
Phase: GSM8K strategies → MATH → StrategyQA → ARC-Challenge
