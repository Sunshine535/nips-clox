# CLOX v2: Auto Review Loop

## Round 1 (2026-04-05)

### Assessment (Summary)
- Score: 2/10
- Verdict: Strong Reject
- Key criticisms:
  - ~90% of experimental claims have no backing data
  - Topology measurements contradict paper's Table 3 (paper claims r̄=0.82, actual is 0.456)
  - CLOX-Adaptive never evaluated
  - Only 1 model (7B), paper claims 3
  - Only 1 seed, paper claims 5
  - No MATH or BBH results
  - No statistical tests computed
  - Cross-model consistency table (Table 5) is fabricated
  - Synthetic DAG experiments never run

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score:** `2/10`
**Verdict:** `No`

As written, this is a strong reject. The issue is not just weak evidence; it is that the paper claims a large experimental program that, by your summary, mostly does not exist. NeurIPS reviewers will forgive a narrow paper. They will not forgive invented breadth.

The current real evidence is also not enough to support the core story. On the only benchmark with a nontrivial signal, GSM8K, `SC-5 = 87.0` and `backward_cloze = 86.0` on `n=200`, which is not a meaningful separation. `targeted_repair = random_repair = 70.5`, so the masking-specific story is not supported there either. StrategyQA is at chance, ARC is saturated, and the claimed GSM8K topology value (`0.82`) is badly contradicted by the actual measurement (`0.456`).

**Ranked Weaknesses And Minimum Fixes**

1. **Disqualifying claim/evidence mismatch** — remove every unrun benchmark, model, condition, seed count, table, ablation, and statistical claim.
2. **Core thesis not empirically demonstrated** — run synthetic DAG + 2-3 real benchmarks.
3. **Topology table unreliable** — regenerate from actual measurements.
4. **Theory-to-observation bridge missing** — validate topology proxy on 30-50 traces.
5. **Evaluation underpowered** — >=500 examples, 3 seeds, paired bootstrap CIs, McNemar.
6. **Adaptive selector absent** — implement and evaluate CLOX-Adaptive.
7. **Benchmark set uninformative** — prioritize benchmarks where methods separate.
8. **Scope inflation** — cut to 1 primary model, 2-3 benchmarks, 4-5 methods.

</details>

### Actions Taken

1. **Synthetic DAG experiments — COMPLETED**
   - Implemented `code/synthetic_dag.py`: 5 graph types × 6 r̄ values × 3 seeds × 2000 trials
   - Theory prediction accuracy: 83.0% across 188 conditions
   - Clear phase transition on tree_b4 at r̄≈0.30
   - Budget-constrained finding: MR dominates chains even at low r̄ (K=3 too weak for SC)
   - Results: `results/synthetic/synthetic_dag_results.json`

2. **Full experiment launched on 2×H100 TP=2 — RUNNING**
   - Model: Qwen/Qwen2.5-32B-Instruct-AWQ
   - Phase 0: Pilot (50 examples × 5 strategies × 4 benchmarks)
   - Phase 1: Topology (200 examples × 5 pilot traces)
   - Phase 2: Full strategies (9 strategies × 3 seeds × 4 benchmarks)
   - Phase 3: Proxy validation (50 examples × [3,5,8,30] pilot traces)

3. **Fixed MATH dataset loader** — benchmarks.py updated for EleutherAI/hendrycks_math

4. **Environment set up** — venv with torch 2.5.1+cu124, vllm 0.6.6, transformers 4.57.6

### Results
- Synthetic DAG: theory validates with 83% accuracy, crossover confirmed
- LLM experiments: in progress (model loaded, pilot phase running)

### Status
- Continuing to Round 2 after experiments complete
- Difficulty: medium

---

## Round 2 (2026-04-05)

### Assessment (Summary)
- Score: 5.5/10 (up from 2/10), projected 7/10 if runs complete
- Verdict: Almost — "plausible but still incomplete"
- Key criticisms:
  1. Real-data validation still unfinished (full n=300 pending)
  2. Topology predicts high-r̄ story well but low-r̄ story poorly
  3. Absolute theory calibration broken (thresholds need revision)
  4. Budget-constrained SC refinement needs formalizing
  5. Topology proxy still untested
  6. CLOX-Adaptive still unevaluated
  7. Single-model framing needed

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score:** `5.5/10` now. `~7/10` if the pending runs land cleanly and the rewrite stays disciplined.
**Verdict:** `Almost`

The paper has moved from "reject for credibility failure" to "plausible but still incomplete." The main danger now is overclaiming from partial alignment.

Key points:
- Real-data validation still unfinished → finish n=300, paired bootstrap, McNemar
- Topology predicts high-r̄ better than low-r̄ → weaken claim or revise predictor
- Absolute calibration broken → recast as ordinal, re-fit thresholds
- Budget-constrained SC → add finite-budget corollary
- Proxy validation → must do, even 30-50 traces
- CLOX-Adaptive → evaluate on held-out split
- Single model acceptable with honest framing

</details>

### Actions Taken (in progress)

1. **Compiled GSM8K partial full results** (300 examples, 3 seeds):

| Strategy | s11 | s23 | s37 | Mean ± Std |
|---|---:|---:|---:|---|
| Standard CoT | 89.33% | 89.33% | 89.33% | 89.33 ± 0.00 |
| Self-Consistency (k=5) | 92.33% | 91.33% | 89.67% | 91.11 ± 1.38 |
| Compute-Matched SC (k=2) | 88.33% | 89.00% | 88.00% | 88.44 ± 0.51 |

2. **Critical topology insight discovered**: ALL 4 benchmarks have short EPL (ℓ/n ≈ 0.24-0.31). We never reach the SC-dominant regime (Theorem 2 requires ℓ ≥ Ω(n)). This explains why repair wins almost everywhere.

Revised narrative:
- Theory establishes two regimes; synthetic DAGs confirm both exist
- Real 32B models operate almost exclusively in the short-ℓ regime
- The key discriminator in practice is r̄, not ℓ
- New threshold τ_r ≈ 0.49 (splits MATH/GSM8K from StrategyQA/ARC)
- StrategyQA breaks pattern due to task structure (boolean composition), not topology

3. **Full experiment continues running** — 9/108 strategy-seed combos done, targeted_repair starting next

### Results
- Round 2 score improved 2→5.5, projected 7 if runs complete
- Key theoretical revision: short-ℓ regime finding changes paper narrative
- Experiment collecting real data for 300 examples × 9 strategies × 3 seeds × 4 benchmarks

### Status
- Experiment running (PID 30709), ~40h remaining
- Waiting for targeted_repair and remaining strategy results
- Paper rewrite pending on full results
- Difficulty: medium
