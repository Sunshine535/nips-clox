# Proof Audit: CLOX Paper — Round 1

**Paper**: CLOX: When Does Inference-Time Reasoning Restructuring Help?
**File**: paper/main.tex (lines 1-759)
**Auditor**: Claude (no cross-model review — Codex MCP unavailable)
**Date**: 2026-04-14
**Difficulty**: nightmare

---

## Summary

| Severity | Count |
|---|---|
| FATAL | 1 |
| CRITICAL | 4 |
| MAJOR | 4 |
| MINOR | 2 |
| **Total** | **11** |

**Acceptance gate: FAIL** (1 FATAL, 4 CRITICAL open)

---

## Issue 1: r̄·(1-ε) repair probability double-counts

- **id**: 1
- **status**: INVALID
- **impact**: GLOBAL
- **severity**: CRITICAL
- **category**: MISSING_DERIVATION / NORMALIZATION_MISMATCH
- **location**: Thm 1 proof, Step 2 (line 625-630)

**Statement**: "Each repaired step succeeds independently with probability r̄(1-ε): factor r̄ for the local context being sufficient, and factor (1-ε) for the re-generation being correct."

**Why invalid**: Definition 3 (line 119-126) defines r_i as:

> r_i = Pr[correct re-generation of v_i | correct values of N(v_i) provided as context]

This is ALREADY the probability of correct regeneration given correct context. It subsumes the model's error rate. The proof then multiplies by (1-ε), interpreting r̄ as a "structural context sufficiency" factor and (1-ε) as "generation accuracy." But that decomposition contradicts Definition 3: r_i is NOT context sufficiency — it IS the full regeneration probability.

If the authors intend r̄ to be a structural property (probability that context provides enough information, independent of model), they need a different definition. Under Definition 3 as written:
- Pr[repair succeeds | context correct] = r_i (not r_i(1-ε))
- Pr[repair succeeds | context partially corrupted] = something smaller (not modeled)

**Counterexample**: Let r_i = 0.9 (model regenerates correctly 90% of the time given correct context) and ε = 0.15. The proof claims repair success = 0.9 × 0.85 = 0.765. But by Definition 3, the correct answer is simply 0.9.

**Affects**: Theorem 1 bound (Eq. 3), all downstream comparisons with CoT and SC.

**Minimal fix**: Either (a) redefine r_i as context sufficiency (task property only, not model-dependent) and add (1-ε) for model error, or (b) use r̄ directly as the repair success probability without the (1-ε) factor. Option (b) is simpler and strengthens the bound.

---

## Issue 2: (1-r̄)^ℓ "absorption" bound is unjustified under deterministic propagation

- **id**: 2
- **status**: UNJUSTIFIED
- **impact**: GLOBAL
- **severity**: CRITICAL
- **category**: LOGICAL_GAP
- **location**: Thm 1 proof, Step 2 (line 631-632)

**Statement**: "The probability that an error in an unrepaired step reaches the answer without encountering a repair barrier is at most (1-r̄)^ℓ (each intermediate step that could absorb the error fails to do so with probability 1-r̄)."

**Why unjustified**: Under the assumed deterministic propagation (φ_ij = 1), an error at step i corrupts ALL downstream steps with probability 1. There is no "absorption" mechanism for unrepaired intermediate steps — that's what deterministic propagation means.

The intended argument seems to be: among the ℓ downstream steps, some may be REPAIRED steps that act as "firewalls." If a repaired step is on the propagation path, it might regenerate correctly despite the corrupted input. But:

1. The proof discusses "unrepaired steps" explicitly, not repaired ones.
2. Even for repaired steps: if the context N(v_i) is corrupted (because the error has propagated to neighbors), Definition 3's r_i does NOT apply — r_i conditions on CORRECT context.
3. Using r̄ as the "absorption probability" per intermediate step conflates two different conditional probabilities: Pr[correct regen | correct context] vs Pr[correct regen | corrupted context].

**Counterexample (candidate)**: Consider a chain of n=10 nodes with ε=0.15, r̄=0.8, ℓ=10. Under deterministic propagation, if step 1 errors, steps 2-10 are ALL corrupted. Even if steps 5-10 are repaired, their context (previous steps) is corrupted, so r_i does not give the repair probability. The actual repair probability in this scenario depends on how well the model handles corrupted context, which is NOT r_i.

**Affects**: The core bound in Theorem 1 (Eq. 3). The second term (n-m)ε(1-r̄)^ℓ is the key term that makes masking favorable — if this term is wrong, the entire masking advantage may not hold.

**Minimal fix**: Introduce a NEW quantity: ρ_i = Pr[correct regen | corrupted context], require ρ_i < r_i, and use (1-ρ_i) instead of (1-r̄) in the propagation term. Alternatively, model repaired steps as barriers explicitly: the probability that an error crosses a repair barrier is (1 - r_i · Pr[barrier's context is correct]).

---

## Issue 3: Independence of repairs is unjustified

- **id**: 3
- **status**: UNJUSTIFIED
- **impact**: LOCAL
- **severity**: MAJOR
- **category**: HIDDEN_ASSUMPTION
- **location**: Thm 1 proof, Step 2 (line 625)

**Statement**: "Each repaired step succeeds independently."

**Why unjustified**: Repairs share context. If step i and step i+1 are both repaired, step i+1's context includes step i's output. If step i's repair fails, step i+1's context is corrupted, affecting its repair probability. The repairs are correlated through shared RCG edges.

**Affects**: The additive structure of the bound (sum of independent per-step contributions).

**Minimal fix**: State explicitly that the bound holds under the assumption that repaired steps are not adjacent (or that they are repaired in topological order with updated context). Alternatively, use a union bound that doesn't require independence.

---

## Issue 4: CoT error model wrong for general RCGs

- **id**: 4
- **status**: OVERSTATED
- **impact**: GLOBAL
- **severity**: CRITICAL
- **category**: SCOPE_OVERCLAIM
- **location**: Thm 1 statement (line 167) and proof Step 1 (line 620-623)

**Statement**: Theorem claims to hold for any "RCG with n steps" but the proof uses:
```
E[err_CoT] = 1 - (1-ε)^n
```

**Why overstated**: This formula is ONLY correct for a chain graph (or any graph where the answer depends on ALL n nodes being correct). For a tree RCG with branching factor b and depth d:
- The answer depends on d ancestors, not all n = b^d nodes
- E[err_CoT] = 1 - (1-ε)^d ≈ dε, NOT nε

For a parallel graph (n independent sub-tasks):
- If the answer requires ALL sub-tasks correct: E[err] = 1-(1-ε)^n (same as chain)
- If the answer requires MAJORITY correct: much lower error

The theorem statement says "Let G be an RCG" (general) but the proof only works for the chain case.

**Counterexample**: Take G as a binary tree of depth 3 (n=7 nodes). The answer node v_7 depends only on the path from root to v_7 (3 steps). E[err_CoT] = 1-(1-ε)^3, not 1-(1-ε)^7. The proof's formula overestimates the CoT error by a factor of ~2.3x, making the masking advantage look larger than it actually is.

**Affects**: All three comparisons in the paper (MR vs CoT, MR vs SC, and the separation theorem).

**Minimal fix**: Either restrict the theorem to chain RCGs, or replace (1-ε)^n with (1-ε)^{d(G)} where d(G) is the depth of the longest path to the answer in G. The latter generalizes correctly.

---

## Issue 5: Theorem 1 proof is chain-specific but claims generality

- **id**: 5
- **status**: OVERSTATED
- **impact**: GLOBAL
- **severity**: MAJOR
- **category**: SCOPE_OVERCLAIM
- **location**: Thm 1 statement (line 167) vs proof (lines 617-655)

**Statement**: Theorem 1 says "Let G be an RCG with n steps" without restricting to chains.

**Why overstated**: Every step of the proof uses chain-specific arguments:
- Step 1: E[err_CoT] = 1-(1-ε)^n (chain formula)
- Step 2: "errors propagate at most ℓ steps before reaching a repair barrier" (linear propagation, not branching)
- Step 4: p = (1-ε)^n as per-trace success probability (chain formula)

For tree/DAG graphs, error propagation is multidimensional, repair barriers have different geometry, and per-trace success depends on graph depth not graph size.

**Affects**: Generality claims in abstract and introduction.

**Minimal fix**: Add "chain" to the theorem statement, or provide separate bounds for different graph classes. The chain restriction is not a serious limitation for the paper's contribution — chain graphs are the natural worst case for EPL.

---

## Issue 6: The masking bound (Eq. 3) is not a coherent probabilistic bound

- **id**: 6
- **status**: UNJUSTIFIED
- **impact**: GLOBAL
- **severity**: CRITICAL
- **category**: LOGICAL_GAP
- **location**: Thm 1, Eq. 3 (line 171)

**Statement**:
```
E[err_MR] ≤ mε(1 - r̄(1-ε)) + (n-m)ε(1-r̄)^ℓ
```

**Why unjustified**: The bound sums contributions from repaired and unrepaired steps. But E[err_MR] = Pr[answer is wrong], which is a SINGLE binary event, not a sum of independent per-step error probabilities.

For this sum to be a valid upper bound, it must be a **union bound**: Pr[answer wrong] ≤ Σ_i Pr[step i contributes an uncorrected error reaching the answer]. This requires:
1. Each term Pr[step i contributes error] is correctly computed (Issues 1-2 contest this)
2. The events are properly defined (what does "reaching the answer" mean for a non-chain graph?)

The proof never establishes that this is a union bound. It just writes the sum as if it were obvious.

Moreover, for a chain: if ANY step has an uncorrected error, the answer is wrong (deterministic propagation). So the events "step i has uncorrected error reaching answer" are NOT disjoint, and their probabilities don't simply add. A union bound would give:

E[err] ≤ Σ_i Pr[step i has unrecoverable error] = mε(1-r̄(1-ε)) + (n-m)ε·1

The (1-r̄)^ℓ factor requires a more sophisticated coupling argument showing that errors at unrepaired steps are blocked by repaired steps — which circles back to Issue 2.

**Affects**: The core bound of the paper.

**Minimal fix**: Explicitly frame as a union bound. For the second term, model the probability that an error at an unrepaired step reaches the answer, accounting for the repair barriers. This requires specifying the spatial distribution of repaired steps (e.g., evenly spaced every n/m steps) and the probability that each barrier successfully blocks the error.

---

## Issue 7: Theorem 2(b) conflates SC (majority voting) with best-of-K (oracle selection)

- **id**: 7
- **status**: INVALID
- **impact**: GLOBAL
- **severity**: **FATAL**
- **category**: IMPLICATION_REVERSAL
- **location**: Thm 2 statement (line 196) and proof (line 670)

**Statement**: "self-consistency with $K$ samples achieves error $\leq \delta$" even when p ≤ 1/2, via "best-of-K selection."

**Why invalid**: Self-consistency is defined (Def 6, line 155) as "$K$ independent traces with majority-vote answer." When p ≤ 1/2, majority voting converges to the WRONG answer as K grows — the majority of traces are wrong, so the majority vote is wrong with probability approaching 1.

The proof switches to "best-of-K selection (choosing the trace whose answer matches any other trace, or the highest-confidence trace)" — this is a DIFFERENT ALGORITHM that requires either:
- A verifier that identifies the correct trace (oracle selection), or
- A confidence scorer (not part of the SC definition), or
- Matching between traces (which only works if correct traces agree — not guaranteed for open-ended questions)

The claim "best-of-K achieves error ≤ (1-p)^K" is the oracle-selection error, not the SC error. For actual SC (majority vote) when p < 1/2:

E[err_SC] → 1 as K → ∞

This REVERSES the theorem's conclusion: SC is WORSE with more samples when p < 1/2, not better.

**Counterexample**: Let ε = 0.15, n = 20, so p = (1-0.15)^20 ≈ 0.039. With K=100 samples:
- Best-of-K (oracle): err ≤ (1-0.039)^100 ≈ 0.019 (excellent)
- Majority vote SC: ≈99.9% of traces are wrong, majority is wrong, err ≈ 1.0 (terrible)

**Affects**: Theorem 2 as stated is FALSE for the p ≤ 1/2 regime. This breaks the "SC dominance" claim, which breaks Theorem 3 (separation), and undermines the paper's main narrative.

**Minimal fix**: Three options:
(a) **Restrict Thm 2 to p > 1/2** (severe limitation — requires n < log(2)/log(1/(1-ε)) ≈ 4.3 for ε=0.15)
(b) **Redefine SC to include answer clustering** (where traces with matching answers form a cluster, and the largest cluster is selected — this is what Wang et al. 2022 actually do, and it works better than pure majority vote)
(c) **Explicitly introduce best-of-K as a separate strategy** in the strategy family Σ, prove the bound for that, and update the narrative.
Option (b) is the most defensible and closest to the existing proof.

---

## Issue 8: Corollary 2 (Adaptive Selector) has no proof

- **id**: 8
- **status**: UNJUSTIFIED
- **impact**: LOCAL
- **severity**: MAJOR
- **category**: UNJUSTIFIED_ASSERTION
- **location**: Cor 2 (line 221-228)

**Statement**: The adaptive selector achieves E[err] ≤ OPT + O(√(log K / M)) + O(ε_est).

**Why unjustified**: No proof is provided. The O(√(log K / M)) term is described as "finite-sample regret of choosing among K strategies" — this resembles a multi-armed bandit explore-then-commit bound, but:
1. The setting is NOT a bandit: there's no reward feedback, just topology estimation.
2. The bound should come from the estimation error of (ℓ̂, r̂) propagated through the error bounds of Thms 1-2. This requires a sensitivity analysis of those bounds to topology mis-estimation.
3. ε_est is never bounded. Without showing ε_est = O(1/√M), the corollary is vacuous.

The √(log K / M) term may come from a Hoeffding bound on the estimated error of each strategy, combined with a union bound over K strategies. But this needs to be spelled out.

**Affects**: The adaptive guarantee claim in the abstract ("within O(√(log K/M)) of oracle").

**Minimal fix**: Provide a proof. The natural approach: show that |err_σ(ℓ̂, r̂) - err_σ(ℓ, r̄)| ≤ L · |(ℓ̂-ℓ, r̂-r̄)| for some Lipschitz constant L (sensitivity analysis), then use concentration of the estimators, then union bound over K strategies.

---

## Issue 9: Theorem 3 "minimax argument" for mixed strategies is hand-waved

- **id**: 9
- **status**: UNJUSTIFIED
- **impact**: LOCAL
- **severity**: MAJOR
- **category**: UNPROVEN_SUBCLAIM
- **location**: Thm 3 proof, last line (line 691)

**Statement**: "This extends to any mixed strategy by a minimax argument."

**Why unjustified**: A mixed strategy that randomizes between MR (with probability q) and SC (with probability 1-q) achieves:

E[err_mixed on D1] = q · err_MR(D1) + (1-q) · err_SC(D1)
E[err_mixed on D2] = q · err_MR(D2) + (1-q) · err_SC(D2)

The minimax gap is:
max(E[err_D1] - OPT_1, E[err_D2] - OPT_2)

For q = 1/2: gap ≈ max(err_SC(D1)/2, err_MR(D2)/2) = Ω(ε)/2 = Ω(ε). OK.

But a strategy that first estimates the topology (spending O(1) pilot traces) and then picks MR or SC accordingly could achieve gap = O(ε_est). This is essentially CLOX-Adaptive, and Cor 2 claims it has gap O(√(log K/M)). So the "no fixed strategy" result is true but the "no strategy whatsoever" extension is false — adaptive strategies can beat fixed ones.

The theorem is stated for "fixed strategy σ ∈ Σ" which excludes adaptive strategies, so technically correct. But the "minimax" comment about mixed strategies is misleading because adaptive ≠ mixed.

**Affects**: Interpretation of the NFL theorem. The result is weaker than it appears.

**Minimal fix**: Remove the "extends to any mixed strategy" comment, or clarify that it applies to strategies that don't observe the task topology (oblivious strategies).

---

## Issue 10: EPL definition ambiguous for multi-path DAGs

- **id**: 10
- **status**: UNCLEAR
- **impact**: COSMETIC
- **severity**: MINOR
- **category**: MISSING_DERIVATION
- **location**: Def 4, Eq. 2 (line 133-136)

**Statement**:
```
ℓ(G) = max_i Σ_{j reachable from i} Π_{(a,b) in path(i,j)} φ_ab
```

**Why unclear**: When there are multiple paths from i to j in a DAG, which path does "path(i,j)" refer to? Options:
- Sum over ALL paths (could be exponential)
- Single shortest path
- Single longest path
- Any path (non-deterministic)

For tree-structured RCGs, path(i,j) is unique, so no ambiguity. For general DAGs, this is undefined.

**Affects**: EPL computation for general RCGs (but paper primarily uses chains/trees where paths are unique).

**Minimal fix**: Either restrict to tree-structured RCGs, or define ℓ using max over all paths (worst-case propagation).

---

## Issue 11: nε approximation validity

- **id**: 11
- **status**: OVERSTATED
- **impact**: COSMETIC
- **severity**: MINOR
- **category**: CONSTANT_DEPENDENCE_HIDDEN
- **location**: Thm 1, Eq. 5 (line 177) and proof Step 1 (line 622)

**Statement**: E[err_CoT] = 1-(1-ε)^n ≈ nε "for small ε"

**Why overstated**: The approximation 1-(1-ε)^n ≈ nε requires nε ≪ 1, not just ε ≪ 1. For the paper's experimental parameters (ε=0.15, n=10), nε = 1.5 and:
- Exact: 1-(0.85)^10 = 0.803
- Approximation: 10 × 0.15 = 1.5 (>1, nonsensical as probability)

**Affects**: Intuition, not formal results (the approximation is only used informally).

**Minimal fix**: Replace "for small ε" with "when nε ≪ 1" or just use the exact formula throughout.

---

## Counterexample Red Team Log

### CE-1: Chain with deterministic propagation breaks (1-r̄)^ℓ (for Issue 2)
- **Setup**: n=5 chain, ε=0.2, r̄=0.8, ℓ=5, m=0 repairs
- **Theorem predicts**: E[err] ≤ 5·0.2·(1-0.8)^5 = 5·0.2·0.00032 = 0.00032
- **Actual** (deterministic propagation, no repairs): E[err] = 1-(1-0.2)^5 = 0.672
- **Ratio**: Actual/predicted = 2100x. The bound is not just loose — it's incorrect.
- **Verdict**: COUNTEREXAMPLE FOUND (algebraically verified). The (1-r̄)^ℓ term is meaningless without repairs.

### CE-2: SC majority vote fails when p < 1/2 (for Issue 7)
- **Setup**: ε=0.15, n=20, p=(0.85)^20≈0.039, K=21
- **SC majority vote**: Needs >10 correct traces. Pr[≥11 of 21 correct] = Σ_{k=11}^{21} C(21,k)·0.039^k·0.961^(21-k) ≈ 10^{-13}
- **Best-of-K oracle**: Pr[all wrong] = (1-0.039)^21 ≈ 0.44
- **Verdict**: COUNTEREXAMPLE FOUND. SC majority vote gives err ≈ 1.0 while oracle gives err ≈ 0.44. Thm 2 claims SC achieves the oracle bound — FALSE.

### CE-3: Attempted counterexample for Thm 1 masking advantage (Issue 4)
- **Setup**: Tree RCG, b=4, depth d=3, n=21 nodes, ε=0.1
- **CoT error (proof claims)**: 1-(1-0.1)^21 ≈ 0.891
- **CoT error (actual, tree)**: 1-(1-0.1)^3 = 0.271 (only 3 ancestors matter)
- **Masking with m=5 repairs**: If repairs target the 3 critical ancestors, err is much lower
- **Verdict**: Proof overestimates CoT error by 3.3x, inflating the masking advantage. CANDIDATE COUNTEREXAMPLE to the theorem's generality.

---

## Decision: FAIL

The paper has **1 FATAL and 4 CRITICAL issues**. The acceptance gate is not met.

### Blocking issues (must fix):
1. **FATAL Issue 7**: Thm 2(b) is false as stated. SC ≠ best-of-K when p < 1/2.
2. **CRITICAL Issue 2**: (1-r̄)^ℓ absorption bound is refuted by CE-1 (2100x error).
3. **CRITICAL Issue 1**: r̄(1-ε) double-counting invalidates Eq. 3.
4. **CRITICAL Issue 4**: General RCG claim but chain-only proof.
5. **CRITICAL Issue 6**: Sum-of-contributions bound lacks union-bound framing.

### Salvage assessment:
The THEORY IS SALVAGEABLE. The core insight (short EPL + high recoverability → masking helps) is likely correct. But the formalization has fundamental errors in the probability model. A rewrite that:
1. Restricts to chain graphs (acceptable for NeurIPS)
2. Fixes the repair probability model (separate context quality from regeneration quality)
3. Models repair barriers explicitly
4. Splits Thm 2 into p>1/2 (SC) and p≤1/2 (oracle/best-of-K)
5. Provides a real proof for Cor 2

would produce a correct paper with the same narrative and contribution.
