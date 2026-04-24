# GPT-5.5 Pro Report Extraction

Source: `GPT55_DIAGNOSIS.md` (root, 1201 lines). Audit of GitHub-visible repo
produced 2026-04-24. This document distills the single execution path — no
synthesis, no recommendation — so subsequent reports can cite line-level anchors.

## Main Method Path (§10, §19)

**CLOX-PCS: Calibrated Portfolio Compute Selection**

> 把 CoT/SC/repair/backward/full-regeneration 等旧策略降级为候选生成器，再用
> held-out calibrated selector 和 value-of-compute gate 在每个 instance 上选择答案
> 或追加计算。

Core missing mechanism: outcome-aware calibrated selection + compute gating.
NOT generic routing; NOT another topology threshold.

## Why the old path failed (§4, §5, §9)

- P02 targeted=random identical across 9 metrics → ablation collapse
- P06 all real benchmarks have short ℓ — theory partition mismatched
- P07 v3/v4 topology values differ sharply — proxy unstable
- P08 CLOXAdaptive collapses to targeted_repair via the `"adaptive"` fallback
- P11 topology correlations ≈ 0 with disagreement/diversity
- P12 BAV agreed=disagreed=80% — agreement gate non-discriminative
- Pilot-to-scale collapse (98% n=50 → 70.5% n=200) on targeted_repair

## Positive signals that survive (§4, §8)

- Oracle-SC gap up to 33.3% on 27B/bbh_logic; 11 cells ≥ 15% → candidate pool has room
- Pilot cross_K5 0.82, oracle_any 0.90, SC 0.88 → candidate complementarity exists
- Low-correlation pairs: 6/28 pairs < threshold; mean abs φ = 0.449 → some independence
- AGD 27B universally non-negative (this repo, 2026-04-24): 4/4 benchmarks +avg 4.2%

## Constraints (§6)

| ID | Rule |
|---|---|
| C01 | Never claim masked repair works without strict slot-filling |
| C02 | No first-N pilot; always seeded split |
| C03 | Topology → features, not hard rule |
| C04 | Exploit oracle gap via selection, not voting |
| C05 | Add calibrated selector + value-of-compute gate |
| C06 | Optimize accuracy–token Pareto, not accuracy alone |
| C07 | Fix metrics before reporting new results |
| C08 | Brier/ECE required, not raw confidence |
| C09 | Numeric + MC + boolean + BBH required |
| C10 | Run minimal tests first |
| C11 | Save every candidate/feature/score/cost/decision |
| C12 | Differentiate from routing/BoN/self-certainty |

## Required task order (§14, §20)

1. Freeze legacy CLOXAdaptive/topology route ✓
2. Immutable result schema + manifests ✓
3. Fix answer extraction + tests ✓
4. Seed reproducibility (SHA256, set_global_seed) ✓
5. Dataset split manifests ✓
6. Portfolio candidate generator ✓
7. Calibrated selector ✓
8. Value-of-compute gate ✓
9. A/B/C evaluation ✓
10. Rewrite paper thesis — **only after minimal tests pass**

## Minimal experiment pre-registration (§15, §20)

- Compile + `pytest tests/test_answer_extraction.py tests/test_seed_reproducibility.py`
- `python code/split_manifest.py --check`
- `python code/calibrated_selector.py debug_tiny` → AUC > 0.55 on toy
- Collect calibration candidates on 50 examples
- Fit selector, report AUC/Brier/ECE
- A/B/C on 100-example test × 3 seeds

## Stop / pivot criteria (§19)

- Stop if selector AUC ≤ 0.55 and no token saving
- Stop if C ≈ B (calibration adds nothing)
- Stop if SC dominates at matched budget
- Stop if answer-metric fixes erase all positive signal
- Pivot if oracle gap high but selector fails → stronger verifier / self-certainty integration
- Pivot if oracle gap disappears after metric fixes → write negative/diagnostic paper

## What must NOT happen (§20)

- No invented alternative method path
- No cherry-picked positive on pilot only
- No silent deletion of negative results
- No metric or dataset change unless explicit
- No treating targeted_repair / BAV / topology routing / cross-vote as final method
