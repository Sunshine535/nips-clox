# CLOX: When Does Inference-Time Reasoning Restructuring Help? A Computational Theory of Optimal Strategy Selection

## Thesis

Inference-time compute scaling has emerged as a powerful paradigm for improving LLM reasoning, but the field lacks a formal understanding of **when** different strategies (chain-of-thought, self-consistency, tree-of-thought, masking/repair) are optimal. We develop the first **computational theory of inference-time strategy selection** grounded in two structural properties of reasoning tasks: **local recoverability** (how much local context suffices to reconstruct masked states) and **error propagation length** (how far early mistakes cascade through the reasoning chain). Our theory yields tight characterizations of optimality, impossibility results for masking-based methods on non-local tasks, and a practical **topology-adaptive strategy selector** that outperforms all fixed strategies.

## Core Claim

> Given a reasoning task characterized by its dependency graph structure, there exists a provably optimal inference-time strategy. Masking/repair outperforms resampling on locally recoverable tasks, self-consistency dominates on error-propagation-heavy tasks, and no single fixed strategy is universally optimal. A topology-aware adaptive selector achieves the best of all worlds.

## Theoretical Framework

### Definition 1: Reasoning Computation Graph (RCG)

A reasoning task instance defines a directed acyclic graph G = (V, E) where:
- V = {v_1, ..., v_n} are intermediate reasoning steps
- E captures dependencies: (v_i, v_j) ∈ E means step j depends on step i
- Each node v_i has a per-step error probability ε_i and a local recoverability score r_i ∈ [0,1]

The **local recoverability** r_i of node v_i is defined as the probability that the correct value of v_i can be recovered from its immediate neighbors alone, conditioned on the rest of the computation being correct.

### Definition 2: Error Propagation Length (EPL)

The EPL of a reasoning graph G is the expected number of downstream nodes affected by a single error at the most vulnerable node:

EPL(G) = max_i E[|{j : j is reachable from i in G, and error at i flips v_j}|]

### Theorem 1: Masking Optimality (Informal)

For tasks where EPL(G) ≤ O(log n) and mean local recoverability r̄ ≥ 1 - δ for small δ, uncertainty-targeted masked repair achieves expected error rate:

E[err_mask] ≤ nε · (1 - r̄ · (1 - ε))

which is strictly better than both single-pass CoT (E[err_CoT] ~ 1-(1-ε)^n) and self-consistency with K samples (E[err_SC] ~ (1-ε)^K · nε) when r̄ is sufficiently high and K is small.

### Theorem 2: Self-Consistency Dominance (Informal)

For tasks where EPL(G) ≥ Ω(n) and r̄ ≤ 1/2, self-consistency with K ≥ log(1/δ)/log(1/(1-ε)) samples achieves error rate ≤ δ, while any single-pass masking strategy cannot reduce the error rate below Ω(nε(1-r̄)^(EPL/2)).

### Theorem 3: No Free Lunch for Fixed Strategies

There exist task distributions D_1, D_2 such that:
- Strategy A (masking) is optimal on D_1 but worst on D_2
- Strategy B (self-consistency) is optimal on D_2 but worst on D_1

Implication: any fixed strategy selection must perform suboptimally on at least one task family.

### Corollary: Adaptive Strategy Selection

A topology-aware selector that estimates (EPL, r̄) from a small number of pilot samples and switches strategy accordingly achieves expected error within O(√(log K / n)) of the oracle-optimal strategy, where K is the number of candidate strategies.

## Method: CLOX-Adaptive

### Phase 1: Task Topology Estimation
Given a reasoning task:
1. Generate M pilot CoT solutions with token-level logprobs
2. Compute per-step uncertainty (entropy), step dependency structure, and confidence patterns
3. Estimate local recoverability r̄ from cross-sample agreement patterns
4. Estimate EPL from error correlation analysis across samples

### Phase 2: Strategy Selection
Based on estimated (EPL, r̄):
- **High r̄, Low EPL** → Uncertainty-Targeted Masked Repair
- **Low r̄, High EPL** → Budget-Matched Self-Consistency
- **High r̄, High EPL** → Hierarchical Masked Repair (mask at bottleneck nodes)
- **Low r̄, Low EPL** → Standard CoT (restructuring unnecessary)

### Phase 3: Execution
Apply the selected strategy with compute-matched token budget.

## Experiments Design

### Benchmarks
1. **GSM8K** (1319 test) - arithmetic, high local recoverability
2. **MATH** (5000 test, Level 1-5) - varying difficulty, mixed topology
3. **StrategyQA** (2290 test) - multi-hop, low local recoverability
4. **ARC-Challenge** (1172 test) - science reasoning, mixed
5. **HumanEval** (164 test) - code generation, high EPL
6. **BIG-Bench Hard** (selected subtasks) - diverse reasoning types

### Models
- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3
- GPT-4o-mini (API, for scaling analysis)

### Conditions (9 total)
1. Standard CoT (zero-shot)
2. Standard CoT (few-shot, 8 examples)
3. Self-Consistency (K=5, budget-matched)
4. Self-Consistency (K=10, doubled budget)
5. Answer-Anchored Backward Cloze Reconstruction
6. Uncertainty-Targeted Selective Masked Repair
7. Random Span Masked Repair (ablation)
8. Full Rationale Regeneration (ablation)
9. **CLOX-Adaptive** (topology-aware selector)

### Evaluation Protocol
- 5 seeds per condition
- Paired bootstrap CI with Bonferroni correction
- Per-example win/loss matrices
- Token-level compute accounting (prompt + completion tokens)
- Latency measurement
- Task topology characterization (EPL and r̄ per benchmark)

### Key Predictions from Theory
| Benchmark | Predicted EPL | Predicted r̄ | Predicted Best Strategy |
|-----------|:---:|:---:|---|
| GSM8K | Low (~2-3) | High (~0.8) | Masked Repair |
| MATH L1-3 | Low-Med | High | Masked Repair |
| MATH L4-5 | High (~8+) | Low (~0.3) | Self-Consistency |
| StrategyQA | High (~5+) | Low (~0.2) | Self-Consistency |
| ARC-Challenge | Medium | Medium | CLOX-Adaptive |
| HumanEval | High (~10+) | Medium (~0.5) | Self-Consistency |
| BBH | Mixed | Mixed | CLOX-Adaptive |

### Ablation Matrix
| Ablation | Tests |
|----------|-------|
| Backward vs Forward fill order | Whether answer-anchoring helps |
| Targeted vs Random masking | Whether uncertainty targeting matters |
| Selective vs Whole repair | Whether locality matters |
| Fixed vs Adaptive strategy | Whether topology estimation helps |
| EPL estimation accuracy | Sensitivity to topology misestimation |
| r̄ estimation accuracy | Sensitivity to recoverability misestimation |
| Pilot samples (1,3,5,10) | How many samples needed for good estimation |

## Quantitative Success Criteria

### Primary (Paper-critical)
- CLOX-Adaptive achieves ≥ 2.0 absolute accuracy improvement over the best fixed strategy averaged across all benchmarks
- Theory correctly predicts which strategy wins on ≥ 4/6 benchmarks
- Masking outperforms SC on ≥ 2 high-recoverability benchmarks by ≥ 1.5 pts

### Secondary
- EPL and r̄ estimates from pilot samples correlate with ground-truth task structure (r ≥ 0.7)
- Adaptive strategy selection overhead is ≤ 20% of total inference budget
- Results replicate across all 3 open models

## Contributions

1. **First computational theory** of inference-time strategy selection, formalizing when masking, self-consistency, and hybrid approaches are optimal via reasoning topology properties (EPL and r̄)
2. **Impossibility and optimality theorems** providing tight characterizations of strategy performance as functions of task structure
3. **CLOX-Adaptive**: a practical topology-aware inference-time strategy selector with provable near-oracle guarantees
4. **Comprehensive empirical validation** across 6+ benchmarks, 4 models, 9 conditions, establishing that:
   - No single fixed strategy dominates
   - Task topology predicts optimal strategy
   - Adaptive selection achieves best-of-all-worlds performance

## Positioning for NeurIPS Best Paper

This work is positioned at the intersection of:
- **Theory** (formal framework with provable guarantees)
- **Systems** (practical algorithm for strategy selection)
- **Empiricism** (comprehensive benchmarking)

It directly addresses the #1 open question in inference-time compute: **when should you use which strategy?** The answer is not "always use the same thing" but "it depends on the task structure, and here's exactly how."

## Risk Mitigation
- If masking never outperforms SC on any benchmark: pivot to "negative result + theory explains why"
- If topology estimation is inaccurate: use oracle topology as upper bound, show gap is small
- If adaptive selector doesn't beat best-fixed: the theory itself is still a major contribution
- All code designed for complete reproducibility with fixed seeds
