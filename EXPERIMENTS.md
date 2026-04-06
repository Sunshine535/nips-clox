# Experiments: CLOX

## 1) Benchmark and Protocol

- **Primary benchmark:** Synthetic seven-segment proxy task
- **Evaluation subset:** 64 examples (stratified across 4 datasets: GSM8K 24, SVAMP 16, StrategyQA 12, AQUA-RAT 12)
- **Seeds:** 5 (11, 23, 37, 47, 59); seeds affect sampling, prompt example order, and tie-breaking
- **Model:** Qwen2.5-1.5B-Instruct or Phi-3-mini-4k-instruct, 4-bit quantized
- **Hardware:** 1× NVIDIA RTX 6000 Ada (49 GB VRAM)
- **Runtime budget:** 300 seconds total (estimated 6s per condition per seed)
- **Decoding:** Temperature 0.2 for deterministic conditions, 0.7 for self-consistency sampling

## 2) Conditions

### Baselines

| Condition | Token Budget | Description |
|---|---|---|
| StandardLeftToRightChainOfThought | ≤96 rationale + 8 answer | Single-pass unrestricted L2R reasoning (Wei et al., 2022) |
| BudgetMatchedSelfConsistencyConsensus | 3 samples × ≤32 each | Majority vote under matched total tokens (Wang et al., 2022) |
| FullRationaleRegenerationRevision | ≤48 first pass + ≤48 revision | Critique + full rewrite (Madaan et al., 2023) |

### Proposed Methods

| Condition | Token Budget | Description |
|---|---|---|
| AnswerAnchoredBackwardClozeReconstruction | 2 candidates + 2 blanks, total ≤96 | Generate answer candidates, then reconstruct rationale backward via masked infilling |
| UncertaintyTargetedSelectiveMaskedRepair | ≤64 first pass + ≤32 repair + ≤8 verify | Entropy-guided selective span repair of uncertain rationale segments |
| RandomSpanMaskedRepair | Matched to uncertainty-targeted | Uniform random span selection (ablation control) |

### Ablations

| Ablation | Parent Method | What Is Removed |
|---|---|---|
| ForwardOrderCloze | AnswerAnchoredBackwardCloze | Backward reconstruction order |
| WholeRationaleMaskRepair | UncertaintyTargetedSelectiveMaskedRepair | Selectivity of local repair |

## 3) Metrics

### Primary
- **final_answer_error_rate:** 100 − exact_match_accuracy (minimize)

### Secondary
| Metric | Definition |
|---|---|
| exact_match_accuracy | Task-standard exact match |
| tokens_per_correct_answer | Total generated tokens / correct predictions |
| latency_per_example_ms | Wall-clock inference time per example |
| budget_normalized_error_reduction | (error_baseline − error_method) / total_generated_tokens |
| selective_repair_ratio | Fraction of rationale tokens modified during repair |
| blank_fill_consistency | Agreement between repaired steps and final answer |
| causal_blank_sensitivity_gap | Answer sensitivity to blank perturbation vs CoT perturbation |
| local_recoverable_subset_error_rate | Error on GSM8K/SVAMP subset |
| non_local_subset_error_rate | Error on AQUA-RAT/StrategyQA subset |
| success_rate | Fraction of seeds completing without failure |

### Statistical Reporting
- 95% bootstrap CI over paired example-level differences
- Wilcoxon signed-rank test across seeds
- Cohen's d on per-seed accuracy
- Per-seed raw values reported

## 4) Main Results

| Method | Accuracy | Error Rate | Notes |
|---|---:|---:|---|
| Standard Left-to-Right CoT | 75.52% | 24.48% | Reference baseline |
| Budget-Matched Self-Consistency | **77.60%** | **22.40%** | Best mean result |
| Full Rationale Regeneration | 76.56% | 23.44% | Slightly above baseline |
| Uncertainty-Targeted Masked Repair | 76.04% | 23.96% | Tied with random masking |
| Random Span Masked Repair | 76.04% | 23.96% | No targeting advantage shown |
| Answer-Anchored Backward Cloze | 75.52% | 24.48% | No gain over baseline |

### Confidence-Split Results
- High-confidence error rate: ~3.125% (all methods perform well)
- Low-confidence error rate: ~53.125% (all methods struggle)

### Seed Variability
| Method | Accuracy Std (across seeds) |
|---|---:|
| Standard CoT | ±1.95 pts |
| Backward Cloze | ±2.66 pts |
| Random Masked Repair | ±3.90 pts |
| Budget-Matched SC | ±4.10 pts |

## 5) Ablation Results

**Critical finding:** Two key ablations produced identical outputs across all 9 metrics:
- Backward cloze vs forward cloze — identical
- Uncertainty-targeted vs whole-rationale masking — identical

This indicates a probable code-path bug where condition-specific logic was not properly triggered. These ablations are **invalid** and cannot support causal claims about backward order or targeted masking.

## 6) Interpretation Against Publication Bar

### Hypothesis 1 (Backward Cloze)
**Not supported.** Answer-anchored backward cloze achieved 75.52%, exactly matching standard CoT. The predicted +2–3 point gain on structured tasks was not observed.

### Hypothesis 2 (Selective Masked Repair)
**Not supported.** Uncertainty-targeted repair (76.04%) did not outperform random masking (76.04%) or budget-matched self-consistency (77.60%). The targeting mechanism showed no advantage.

### Hypothesis 3 (Efficiency)
**Partially explored.** The pilot did not include full latency/token accounting to evaluate efficiency claims rigorously.

### Overall Assessment
- Quality gate: **2.3/4.0 (degraded mode)**
- Decision: **REFINE** (not PROCEED, not full PIVOT)
- The result is classified as **negative-but-informative**

## 7) Statistical Checks

### Effect-Size Analysis
- 64 examples means 1 error = 1.5625 percentage points
- The best-vs-baseline gap (2.08 pts) corresponds to ~1.33 examples
- This is too coarse for confident ranking of close methods

### Variance vs Mean Differences
- Self-consistency std (4.10 pts) exceeds its advantage over baseline (2.08 pts)
- Ranking instability is a serious concern at this sample size

### Missing Paired Inference
No paired bootstrap CI, McNemar test, permutation test, or per-example win/loss tables were computed. All methods ran on the same examples, so paired tests are required for credible comparison.

## 8) Methodology Gaps

1. **Repair and validate ablations** — unit tests for condition-specific behavior, prompt/mask/trace dumps
2. **Use real reasoning benchmarks** — GSM8K, SVAMP, MultiArith, MATH subsets
3. **Report true compute parity** — prompt tokens, completion tokens, total tokens, number of calls, latency
4. **Add paired statistical analysis** — per-example win/loss matrices, bootstrap or permutation tests, CIs
5. **Test the mechanism directly** — step-level error correction rates, wrong→right transition analysis
