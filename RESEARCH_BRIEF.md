# Research Brief: CLOX v3 — Revolutionary Inference-Time Compute

## Target
NeurIPS 2025 best paper. Must be genuinely revolutionary, not incremental.

## Hardware
- 2x NVIDIA H100 80GB HBM3
- vLLM 0.19.0 infrastructure already operational
- Auto multi-GPU TP support required

## What We Learned (CLOX v2 — Lessons from Failure)

### The Core Idea That Partially Worked
Reasoning traces have measurable structural properties:
- **Local recoverability (r-bar)**: probability a masked step can be reconstructed from context
- **Error propagation length (ell)**: how far errors cascade through reasoning chains
- These ARE measurable and DO vary across tasks (confirmed on 4 benchmarks, 200 examples each)

### What Failed
1. **Theory-practice gap**: Theory predicts two clear regimes (short-ell favors repair, long-ell favors SC). Reality: ALL benchmarks have short ell (1.2-1.6). The SC-dominant regime never appears in practice.
2. **Pilot-to-scale collapse**: targeted_repair went from 98% (n=50) to 70.5% (n=200) on GSM8K. Overfitting to easy examples.
3. **Strategy indistinguishability**: At scale, SC-5 (87%) dominates everything on GSM8K. On StrategyQA, everything is at chance (50%). On ARC, everything saturates (89-90%).
4. **Review trajectory**: 2/10 → 5.5/10 after two rounds. Core claim weakened to "negative but informative."

### What's Salvageable
- vLLM engine wrapper (battle-tested, auto-GPU detection)
- Topology estimation code (produces real r-bar and ell from pilot traces)
- Benchmark loaders (GSM8K, MATH, StrategyQA, ARC-Challenge)
- Statistical evaluation framework (paired bootstrap, McNemar, Cohen's d)
- Understanding of the inference strategy landscape

## Research Direction Constraints

### Must Have
- Builds on inference-time compute / reasoning — the hottest area in ML
- Strong theoretical grounding with matching empirical results
- Uses latest models (Qwen3-32B or newer, not legacy models)
- All experiments reproducible on 2xH100
- Multi-seed, proper statistical tests, no overclaiming
- Genuinely novel — not "yet another benchmark study"

### Must NOT
- Be a negative result paper (those don't win best paper)
- Require >2 GPU for core experiments
- Depend on proprietary API access (OpenAI, Anthropic)
- Simply add more baselines to a known comparison
- Be incremental over existing work (Snell et al. 2024, Wang et al. 2022, etc.)

## Key Open Questions in the Field (as of 2026)

1. **When to spend more compute at test time?** Snell et al. 2024 showed compute-optimal allocation depends on problem difficulty, but there's no principled theory for HOW to allocate across strategies.

2. **Can LLMs self-correct without external signals?** Huang et al. 2024 showed NO — but what if structural signals (like our topology metrics) serve as the external signal that breaks this barrier?

3. **What determines reasoning quality?** We know CoT helps, but WHY do some problems benefit from SC while others benefit from repair? The structural answer is missing.

4. **Compute-optimal inference scaling laws** — We have scaling laws for training but NOT for inference. What's the right way to scale test-time compute?

5. **Adaptive inference** — How to dynamically adjust compute per-instance rather than using fixed budgets?

## Desired Paper Shape
- 1-2 clean theorems that make predictions
- Predictions CONFIRMED by experiments (not contradicted)
- Clear practical impact (people can use this)
- Simple, elegant core idea (not a kitchen sink of 9 strategies)
- Changes how the community thinks about test-time compute
