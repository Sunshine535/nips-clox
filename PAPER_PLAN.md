# Paper Plan: CLOX — Partial Masking as a Control for Synthetic Reasoning

## Title

**CLOX: Partial Masking as a Control for Synthetic Reasoning**

## Positioning Decision

The current evidence does **not** support the strongest version of the headline claim. The paper should be framed as:

> A careful empirical study of partial masking at inference time, rather than a triumphant method paper claiming a clear win.

Central message:
- Partial masking and cloze-style restructuring are **feasible** inference-time controls.
- They do **not automatically improve** reasoning.
- In this pilot, **self-consistency outperformed masking-specific variants**.
- The field should treat masking as a **hypothesis to test**, not an established improvement mechanism.

## Core Claim

CLOX is a unified family of partial-masking and repair operators for LLM reasoning. On a synthetic seven-segment proxy benchmark, masking-based methods remain competitive with standard CoT but do not outperform it reliably; the best mean result (0.1927 ± 0.0195 exact-match) comes from uncertainty-targeted selective masked repair, versus 0.1615 for self-consistency and 0.0885 for the standard L2R baseline.

## One-Paragraph Thesis

This paper asks whether large language models can benefit from partial masking at inference time—by reconstructing a rationale in cloze form, selectively repairing uncertain spans, or conditioning reasoning on answer anchors. We introduce the CLOX framework, a unifying view of inference-time masking and rationale reconstruction, and compare backward cloze reconstruction, selective masked repair, random masking, full rationale regeneration, and budget-matched self-consistency. On a pilot seven-segment reasoning benchmark, masking-based methods remain competitive with standard chain-of-thought but do not outperform it reliably; the best mean result comes from self-consistency rather than the flagship cloze method. The paper therefore contributes a more precise conclusion: inference-time restructuring appears viable, but masking-specific gains remain unproven.

## Claims → Evidence Matrix

| # | Claim | Evidence | Section |
|---|-------|----------|---------|
| C1 | Inference-time partial masking is feasible for decoder-only LLMs | All masking conditions execute without collapse; accuracy stays within baseline range | §4.1 |
| C2 | Backward cloze does not outperform standard CoT | Backward cloze 75.52% = Standard CoT 75.52% (identical) | §4.2 |
| C3 | Self-consistency outperforms all masking variants | SC 77.60% vs best masking 76.04% | §4.2 |
| C4 | Uncertainty targeting does not beat random masking | Both at 76.04%; no targeting advantage | §4.3 |
| C5 | Confidence split reveals sharp performance stratification | High-conf error ~3.1%, low-conf error ~53.1% | §4.4 |
| C6 | Key ablations are invalid, blocking causal claims | Backward/forward cloze and targeted/whole masking produce identical outputs | §4.3 |

## Section Structure

### Abstract (~190 words)
- Problem: LLM reasoning is strictly left-to-right; unclear whether partial masking can improve it.
- Method: CLOX framework with 5 inference-time operators.
- Key numbers: Standard CoT 75.52%, backward cloze 75.52%, self-consistency 77.60%, masked repair 76.04%.
- Takeaway: masking is feasible but does not demonstrate advantage over baseline.

### §1 Introduction (~900 words)
- §1.1 Motivation: autoregressive reasoning commits early and may propagate mistakes.
- §1.2 Gap: partial masking during inference for causal LLM reasoning is under-tested.
- §1.3 Approach: introduce CLOX as a framework for inference-time masking operators.
- §1.4 Main finding: negative-but-informative — masking is viable but not advantageous.
- §1.5 Contributions: (1) unifying formulation, (2) controlled pilot comparison, (3) negative result, (4) methodological recommendations.

### §2 Related Work (~700 words)
- §2.1 Inference-time reasoning: CoT, self-consistency, Tree of Thoughts, least-to-most.
- §2.2 Masked modeling and infilling: BERT, GLM, fill-in-the-middle, span corruption.
- §2.3 Self-correction and revision: self-refine, critique-revise, plan-and-solve.
- Each subsection ends with how CLOX differs.

### §3 Method (~1200 words)
- §3.1 Problem formulation: inference-time operator family over reasoning traces.
- §3.2 Answer-anchored backward cloze: answer candidates → sparse scaffold → backward infill.
- §3.3 Uncertainty-targeted selective repair: generate → entropy scan → mask → repair.
- §3.4 Ablation operators: random masking, full regeneration, forward cloze.
- §3.5 Evaluation protocol: compute matching, token accounting, paired comparison.

### §4 Experiments (~1500 words)
- §4.1 Setup: benchmark, model, baselines, metrics, seeds.
- §4.2 Main results: Table 1 — all conditions vs baselines.
- §4.3 Ablation analysis: targeting vs random, backward vs forward, selective vs whole.
- §4.4 Confidence-split analysis: high/low confidence error rates.
- §4.5 Statistical assessment: effect sizes, variance, power analysis.

### §5 Discussion and Conclusion (~500 words)
- Why the negative result is informative.
- What the result does and does not rule out.
- Methodological lessons for future inference-time masking studies.
- Future directions: real benchmarks, larger models, repaired ablations.

### Appendix
- A: Full per-seed results for all conditions.
- B: Prompt templates and example outputs.
- C: Implementation details and reproducibility.

## Figures and Tables

| ID | Type | Content |
|----|------|---------|
| Fig 1 | Teaser | CLOX framework diagram — scaffold with blanks, repair loop, answer extraction |
| Table 1 | Main results | All 6 conditions: accuracy, error rate, token count |
| Table 2 | Ablation | Backward/forward cloze, targeted/random/whole masking |
| Table 3 | Confidence split | High/low confidence error rates per method |
| Fig 2 | Bar chart | Method comparison with error bars (seed std) |
| Fig 3 | Schematic | Detailed method pipeline for backward cloze and selective repair |

## Key Numbers

### Main Comparison (64 examples, 5 seeds)
- Standard CoT: 75.52% (±1.95)
- Backward Cloze: 75.52% (±2.66)
- Self-Consistency: **77.60%** (±4.10)
- Masked Repair (targeted): 76.04% (±3.90)
- Masked Repair (random): 76.04% (±3.90)
- Full Regeneration: 76.56%

### Confidence Stratification
- High-confidence error: ~3.125%
- Low-confidence error: ~53.125%
