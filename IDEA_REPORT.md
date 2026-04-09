# Idea Discovery Report (v2)

**Direction**: Revolutionary inference-time compute optimization for LLM reasoning
**Date**: 2026-04-09
**Pipeline**: research-lit → idea-creator → novelty-check → research-review

## Executive Summary

After surveying 60+ papers, generating 8 ideas, running 5 novelty checks, and 1 external review (4/10), the inference-time compute field is extremely competitive in 2025-2026. Our top idea (entropy trajectory) was scooped. Our second idea (structural verification) was demolished by review because our own data contradicts it. Two viable directions remain that leverage our existing 9-strategy infrastructure.

## Literature Landscape

### The Field (2025-2026)
- **Test-time compute scaling**: Established by Snell2024, extended by Wu2024, Liu2025, Roberts2026
- **Entropy for reasoning**: SATURATED (12+ papers including Zhao2026, EAS2025, DiffAdapt2025)
- **Reasoning topology**: Active but analysis-only (Shape of Reasoning, Molecular Structure)
- **Strategy selection**: Growing (DiffAdapt, DOTS, Route-to-Reason, BoM)
- **Process rewards**: Expensive (ThinkPRM, PAVs, GenPRM)

### Key Gap That Remains
**Nobody has shown that STRATEGY-LEVEL diversity (not just sample-level diversity) improves test-time compute.** All SC/BoN work samples from ONE strategy. Cross-strategy approaches (FOBAR, CoTnPoT) are limited to K=2 math-specific pairs.

## Eliminated Ideas

### idea:001 — CLOX v2 Topology → Strategy [FAILED]
- Experimental data contradicts core thesis at scale
- SC dominates regardless of topology profile
- Review trajectory: 2/10 → 5.5/10

### idea:002 — Entropy Trajectory Scaling Law [SCOOPED]
- Zhao 2026 (arXiv:2603.18940) is borderline identical
- 12+ papers cover all sub-claims

### idea:003 — Zero-Cost Structural Verification [REVIEWED: 4/10]
- Theorem unprovable (faithfulness assumption false)
- Own data: SC-5 (87%) >> structural methods (70.5%)
- "Zero-cost" is dishonest

### idea:004 — Late-Stage Fragility Theory [PARTIAL, LOW IMPACT]
- ASCoT already named and demonstrated the phenomenon
- Only the manifold-width theory is novel — thin contribution alone

### idea:007 — Entropy-Gated Adaptive Depth [CROWDED]
- Many papers in this space (s1, L1, SelfBudgeter)

### idea:008 — Implicit Verifier Hypothesis [CLOSE TO Self-Certainty]
- Self-Certainty (NeurIPS 2025) already does training-free BoN scoring

## Surviving Ideas

### IDEA A: Cross-Strategy Verification + Selection-Voting Theory
**Title**: "Strategy Diversity Breaks the Voting Ceiling: Cross-Strategy Inference for Compute-Optimal Reasoning"

**Thesis**: Standard self-consistency generates K samples from ONE strategy, creating correlated errors. Cross-strategy inference — running different strategies (CoT, backward cloze, targeted repair, full regeneration) — produces structurally diverse perspectives with approximately independent errors. We prove this independence condition determines when cross-strategy voting beats single-strategy SC, derive the crossover boundary, and demonstrate 15-30% compute savings on diverse benchmarks.

**Why novel**:
- General K-strategy framework with ε-independence analysis (NEW)
- No existing work goes beyond K=2 or beyond math (FOBAR, CoTnPoT)
- Crossover K* = f(pass_rate, strategy_diversity) as practitioner decision rule (NEW)
- Topology-based strategy diversity measurement (NEW)

**Theoretical contributions**:
1. Strategy Independence Theorem: If K strategies have pairwise error correlation ρ < 1/(2K-1), cross-strategy voting achieves error O(ε^K) vs O(ε^(K/2)) for single-strategy SC
2. Crossover Theorem: There exists K* below which any scorer with AUC > 0.5+ε outperforms majority voting, and K* increases with problem difficulty
3. Portfolio Optimization: Derive the optimal strategy subset as a function of topology (r-bar, ε-independence)

**Experiments**:
1. Run all 9 strategies on 4 benchmarks × 200 examples × 3 seeds
2. Measure pairwise error correlation between strategies (is ε-independence real?)
3. Cross-strategy vote vs SC at matched token budgets (K=3,5,9)
4. Crossover analysis: vary K and difficulty, find the boundary
5. Portfolio optimization: which subset of strategies maximizes accuracy/token?
6. Topology as diversity predictor: does r-bar predict which strategy portfolio works?

**Feasibility**: EXCELLENT. All 9 strategies implemented in strategies_v2.py. vLLM engine ready. Benchmarks loaded. ~5-7 GPU days on 2xH100.

**Impact potential**: 7-8/10. If cross-strategy voting beats SC at matched budget, it's immediately actionable and changes how practitioners think about test-time compute.

**Risk**: MEDIUM. The key risk is that strategies share failure modes (same model, correlated errors). If ε-independence doesn't hold, the theory collapses. Need pilot to validate.

---

### IDEA B: Pure Theory — The Selection-Voting Crossover
**Title**: "When Selection Beats Voting: A Unified Theory of Trace Quality Scoring for Test-Time Compute"

**Thesis**: Derive the precise boundary where trace selection outperforms majority voting as a function of (pass_rate, scorer_AUC, K).

**Why novel**: The specific K* formula doesn't exist. BoM (ICLR 2026) gives minimax bounds but not per-instance.

**Risk**: Straightforward theory, may not impress. Score projection: 6-7/10.

---

## Recommendation

**IDEA A (Cross-Strategy Verification)** is the recommended direction because:
1. It directly leverages ALL of CLOX v2's infrastructure (9 strategies, topology metrics, benchmarks)
2. It makes a falsifiable prediction (cross-strategy > single-strategy) testable in days
3. It builds on a real gap (nobody has done general K-strategy comparison)
4. It's practically useful (changes how people allocate test-time compute)
5. It subsumes Idea B (the theory applies within the cross-strategy framework)

**Critical first experiment**: Before committing, validate the core premise. Run all 9 strategies on 50 GSM8K problems and compute pairwise error correlation. If ρ < 0.3 between at least 3 strategy pairs, proceed. If ρ > 0.7 for all pairs, abandon and consider Idea B as standalone.

## Next Steps
- [ ] Pilot validation of cross-strategy independence
- [ ] Full experiment deployment on 2xH100
- [ ] Theory formalization
- [ ] /auto-review-loop for paper refinement
