# Proof Skeleton: CLOX Paper

## 1. Dependency DAG

```
Def 1 (RCG) ──────────────┐
Def 2 (Error Process) ─────┤
Def 3 (Local Recoverability)┤
Def 4 (EPL) ───────────────┤
Def 5 (Token Budget) ──────┤
Def 6 (Strategy Family) ───┤
                            ▼
                    Thm 1 (Masking Advantage)
                            │
                            ▼
                    Thm 2 (Resampling Advantage) ──uses──> Hoeffding's inequality
                            │
                            ▼
                    Thm 3 (Strategy Separation) ──uses──> Thm 1 + Thm 2
                            │
                    ┌───────┴────────┐
                    ▼                ▼
            Cor 1 (Budget SC)   Cor 2 (Adaptive) ──uses──> "bandit regret" [UNVERIFIED]
```

**Cycles detected**: NONE (DAG is acyclic)
**Forward references**: Thm 3 proof uses Thm 1 and Thm 2 (valid)

## 2. Assumption Ledger

### Theorem 1 (Masking Advantage)
| Hypothesis | Where verified |
|---|---|
| RCG with n steps | Stated |
| Uniform per-step error ε < 1/2 | Stated |
| Mean local recoverability r̄ | Stated |
| EPL = ℓ | Stated |
| **Deterministic propagation (φ_ij = 1)** | Stated |
| Token budget B = n + mc | Stated |
| Entropy-based targeting policy π | Stated but **not used in proof** |
| **Chain graph structure** | **UNSTATED — used silently in proof** |
| **Repairs are independent** | **UNSTATED — used silently in proof** |
| **r̄ represents context sufficiency, not regeneration probability** | **INCONSISTENT with Def 3** |

### Theorem 2 (Resampling Advantage)
| Hypothesis | Where verified |
|---|---|
| ℓ ≥ Ω(n), r̄ ≤ 1/2 | Stated |
| Same model as Thm 1 | Stated |
| **p > 1/2 for Hoeffding case** | Stated (Case a) |
| **p ≤ 1/2 uses best-of-K, not majority voting** | **UNSTATED — proof switches algorithm** |
| **Best-of-K requires verifier/oracle** | **UNSTATED** |

### Theorem 3 (Strategy Separation)
| Hypothesis | Where verified |
|---|---|
| Fixed strategy σ ∈ Σ | Stated |
| Budget B = 5n | Stated |
| **Extension to mixed strategies** | **CLAIMED but not proven ("minimax argument")** |

### Corollary 2 (Adaptive Selector)
| Hypothesis | Where verified |
|---|---|
| M pilot samples | Stated |
| Estimators from Section 3.4 | Stated |
| **O(√(log K / M)) regret bound** | **NO PROOF given** |
| **ε_est bounded** | **NO BOUND given** |

## 3. Typed Symbol Table

| Symbol | Type | Definition | Issues |
|---|---|---|---|
| G = (V, E) | DAG, V = {v_1,...,v_n} | Reasoning computation graph | OK |
| ε_i | scalar ∈ [0,1] | Per-step error given correct parents | OK |
| ε | scalar ∈ (0, 1/2) | max_i ε_i (worst-case) | OK |
| φ_ij | scalar ∈ [0,1] | Error propagation probability | OK |
| r_i | scalar ∈ [0,1] | Pr[correct regen \| correct N(v_i)] | **CONFLICT: proof reinterprets as "context sufficiency"** |
| r̄ | scalar ∈ [0,1] | (1/n) Σ r_i | **Meaning drifts between Def 3 and Thm 1 proof** |
| ℓ | scalar ∈ [1,n] | EPL: max_i Σ_j Π_{path} φ_ab | **Ambiguous for multi-path DAGs** |
| B | integer > 0 | Token budget | OK |
| K | integer > 0 | Number of SC samples | OK |
| m | integer ∈ [0,n] | Number of repaired steps | OK |
| c | integer > 0 | Per-step repair cost | OK |
| p | scalar ∈ [0,1] | Per-trace success = (1-ε)^n | **Only valid for chains** |
| π | function [n] → [0,1] | Targeting policy | Stated but unused in proofs |

## 4. Canonical Quantified Statements

### Theorem 1
```
∀ chain G with n nodes, ∀ε ∈ (0, 1/2), ∀r̄ ∈ [0,1], ∀ℓ ∈ [1,n] with φ_ij = 1:
  E[err_MR] ≤ mε(1 - r̄(1-ε)) + (n-m)ε(1-r̄)^ℓ

  Furthermore, ∃δ_0 > 0 such that when ℓ ≤ C log n and r̄ ≥ 1 - δ for δ ≤ δ_0/n:
    ∃m = Θ(n): E[err_MR] < E[err_CoT] AND E[err_MR] < E[err_SC_K] ∀K ≤ B/n
```
**Issue**: Stated for general RCG but only proven for chains. δ_0 dependence on (ε, c, C) not specified.

### Theorem 2
```
∀ chain G with n nodes, ∀ε ∈ (0,1/2), r̄ ≤ 1/2, ℓ ≥ Ω(n):
  (a) ∀ MR_{m,π} with m ≤ n: E[err_MR] ≥ ε(1 - r̄^{⌊ℓ/2⌋})
  (b) When p = (1-ε)^n > 1/2: SC_K with K ≥ 2log(1/δ)/(2p-1)² achieves err ≤ δ
      When p ≤ 1/2: BEST-OF-K [NOT SC] achieves err ≤ (1-p)^K
```
**Issue**: Part (b) p ≤ 1/2 switches from SC to best-of-K without declaring the algorithm change.

## 5. Micro-Claim Inventory

### MC-1: CoT error = 1-(1-ε)^n
Context: Chain graph, uniform ε, deterministic propagation
⊢ Goal: E[err_CoT] = 1-(1-ε)^n
Rule: Answer correct iff ALL n steps correct; independence
Side-conditions: **Steps must be serial (chain) — NOT stated for general RCG**

### MC-2: Repair success probability = r̄(1-ε)
Context: Step i selected for repair, context N(v_i) provided
⊢ Goal: Pr[repair succeeds] = r̄(1-ε)
Rule: r̄ for context sufficiency × (1-ε) for correct generation
Side-conditions: **CONFLICT with Def 3: r_i already IS the regeneration probability given correct context. Multiplying by (1-ε) double-counts.**

### MC-3: Propagation absorption = (1-r̄)^ℓ
Context: Error at unrepaired step, deterministic propagation φ=1
⊢ Goal: Pr[error reaches answer] ≤ (1-r̄)^ℓ
Rule: Each intermediate step "absorbs" with probability r̄
Side-conditions: **UNJUSTIFIED. Under deterministic propagation, errors propagate with certainty. "Absorption" requires repaired steps on the path, but proof discusses UNREPAIRED steps.**

### MC-4: SC comparison for growing n
Context: K = B/n samples, p = (1-ε)^n
⊢ Goal: err_SC → 1 for fixed K as n → ∞
Rule: Each trace wrong w.p. 1-p → 1
Side-conditions: Requires K fixed. If K grows with B, this fails. **K ≤ B/n is stated but B's growth with n is unspecified.**

### MC-5: Masking lower bound via critical path
Context: Node i* with ℓ downstream nodes, r̄ ≤ 1/2
⊢ Goal: E[err_MR] ≥ ε(1 - r̄^{⌊ℓ/2⌋})
Rule: Repair each of ⌊ℓ/2⌋ nodes independently
Side-conditions: **Why ⌊ℓ/2⌋ and not ℓ? The factor 1/2 is unexplained.**

### MC-6: Best-of-K error = (1-p)^K
Context: K independent traces, p = per-trace success
⊢ Goal: Pr[all K wrong] = (1-p)^K
Rule: Independence + complement
Side-conditions: **Requires oracle selection (know which trace is correct). Standard SC majority vote does NOT achieve this when p < 1/2.**

### MC-7: Thm 3 minimax extension
Context: Fixed σ suboptimal on at least one of D1, D2
⊢ Goal: Any mixed strategy also suboptimal
Rule: "minimax argument"
Side-conditions: **No proof given. A strategy that randomizes MR vs SC with probability depending on topology estimation could potentially avoid the Ω(ε) gap.**

## 6. Limit-Order Map

| Statement | Limit | Uniform over | Issues |
|---|---|---|---|
| E[err_CoT] ≈ nε | ε → 0 | Fixed n | **Only valid when nε ≪ 1** |
| (1-r̄)^ℓ = o(1/n) | n → ∞ | Fixed ε, r̄ = 1-O(1/n), ℓ = O(log n) | OK if constants compatible |
| p = (1-ε)^n → 0 | n → ∞ | Fixed ε | OK |
| err_SC → 1 for fixed K | n → ∞ | Fixed K, ε | OK |
| O(√(log K / M)) regret | M → ∞ | Fixed K | **No proof; likely requires sub-Gaussian topology estimates** |
| O(ε_est) sensitivity | ε_est → 0 | ? | **ε_est never bounded; uniformity scope unknown** |
