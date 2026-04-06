# CLOX v2: Idea Discovery Report

## Selected Direction: Topology-Aware Compute-Optimal Inference

### Core Insight

Reasoning traces have measurable structural properties — **local recoverability** (r̄) and **error propagation length** (ℓ) — that determine which inference-time strategy is optimal. A practical topology estimator that measures these properties from cheap pilot traces enables compute-optimal strategy routing: achieving self-consistency-level accuracy at 40-60% of the compute cost.

### Why This is Novel (Gap Analysis)

| Area | State of the Art | Open Gap | CLOX Fills |
|------|-----------------|----------|------------|
| Test-time compute (Snell et al. 2024) | Difficulty-based routing | No *structural* predictor for compute allocation | Topology metrics as structural features |
| Process rewards (Lightman et al. 2023) | Trained step-level verifiers | Requires expensive annotations | Training-free structural signal from pilot traces |
| Graph of Thoughts (Besta et al. 2024) | DAG *generation* structure | No *diagnostic* topology of existing traces | Measurable properties (r̄, ℓ) of natural traces |
| Self-correction (Huang et al. 2024) | Models can't self-correct without external signal | Need external signal for repair | Topology metrics = external structural signal |

### Theoretical Framework

**Theorem 1 (Masking Advantage):** When ℓ ≤ O(log n) and r̄ ≥ 1−δ, targeted masked repair achieves lower error than both CoT and SC.

**Theorem 2 (Resampling Advantage):** When ℓ ≥ Ω(n) and r̄ ≤ 1/2, self-consistency is provably preferable to any masking strategy.

**Theorem 3 (No Free Lunch):** No fixed strategy dominates across both regimes.

**Corollary (Adaptive Optimality):** CLOX-Adaptive, which estimates (r̄, ℓ) from M pilot traces and selects accordingly, achieves error within OPT + O(√(log K / M)).

### Experimental Design

**Models:**
- Primary: Qwen3-32B (latest, 2×H100 TP=2)
- Secondary: Qwen2.5-72B-Instruct-AWQ

**Benchmarks:**
- GSM8K (1319 examples, arithmetic reasoning)
- MATH L1-3 (structured math)
- StrategyQA (multi-hop boolean reasoning)
- ARC-Challenge (science reasoning)

**Strategies (9):**
1. Standard CoT (baseline)
2. Self-Consistency K=8
3. Compute-Matched SC K=2
4. Uncertainty-Targeted Repair (entropy-guided)
5. Random Repair (ablation control)
6. Backward Cloze (answer-anchored backward reconstruction)
7. Full Regeneration (critique + rewrite)
8. Hierarchical Repair (bottleneck-aware)
9. CLOX-Adaptive (topology-guided selection)

**Evaluation:**
- 5 seeds per condition
- Paired bootstrap CI, McNemar test, Cohen's d
- Exact token accounting for compute fairness
- Per-example topology profiling

### Target Claims

1. **Topology varies across tasks**: GSM8K has higher r̄ (local arithmetic) vs StrategyQA has higher ℓ (multi-hop chains)
2. **Topology predicts strategy performance**: In the (r̄ ≥ 0.65, ℓ ≤ log n) regime, repair beats SC; reversed in (r̄ ≤ 0.45, ℓ ≥ n/2)
3. **Adaptive routing is compute-optimal**: CLOX-Adaptive matches SC accuracy at 40-60% compute cost
4. **Phase transitions are sharp**: Strategy dominance switches rapidly at topology boundaries

### Risk Assessment

- **Risk**: Topology metrics may not be sufficiently predictive with real LLM traces (vs theoretical assumptions)
  - **Mitigation**: Use 8 pilot traces for robust estimation; test correlation empirically
- **Risk**: Repair may not outperform SC even in favorable topology regime
  - **Mitigation**: This is a valid negative result — report it as evidence for Theorem 2's dominance
- **Risk**: Compute overhead of topology estimation negates savings
  - **Mitigation**: Batch topology estimation + amortize across similar problems

### Status

- [x] Code implementation complete (engine.py, strategies_v2.py, topology_v2.py, run_clox.py)
- [x] Smoke test passed (7B model, GSM8K)
- [ ] Qwen3-32B download in progress
- [ ] Phase 1: Topology characterization
- [ ] Phase 2: Strategy comparison
- [ ] Phase 3: Adaptive routing evaluation
- [ ] Paper writing

### Pilot Signal

From the 7B smoke test:
- Topology estimation works: r̄=0.561, ℓ=1.07 for first GSM8K example
- All 5 tested strategies produce distinct outputs (no ablation collapse)
- Backward cloze now actually performs backward reconstruction
- Real entropy from top-20 softmax distribution (not fake single-token entropy)
