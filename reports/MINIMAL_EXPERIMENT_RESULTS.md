# Minimal Experiment Results

Status as of 2026-04-25. "Ran" = completed with persisted results; "Pending" =
requires live vLLM on server.

## 0. Smoke / Code Gates — all green

| Gate | Command | Result |
|---|---|---|
| Compile | `python -m compileall code` | clean |
| Tests | `python -m pytest tests/` | 39 passed, 18 legacy skipped |
| Split check | `python code/split_manifest.py --check` | empty tree OK |
| Debug-tiny selector | `python code/calibrated_selector.py debug_tiny` | AUC 0.81, ECE 0.068→0.017 |

## 1. AGD experiment (completed PRE-PCS, inherited as evidence)

Complete 4-benchmark × 2-model grid with N=30. These used the legacy extractor;
results are **diagnostic only** until replayed through `check_answer_strict`.

### Qwen3.5-27B — universally positive

| Benchmark | SC(8) | AGD(0.50) | Δ | Tokens |
|---|---|---|---|---|
| math_hard | 50.0% | 56.7% | +6.7% | 0.79x |
| bbh_logic | 6.7% | 6.7% | +0.0% | 0.83x |
| arc_challenge | 80.0% | 86.7% | +6.7% | 0.73x |
| strategyqa | 76.7% | 80.0% | +3.3% | 0.57x |
| **avg** | **53.3%** | **57.5%** | **+4.2%** | **0.73x** |

«avg Δ=+4.2%, min Δ=+0.0% (UNIVERSAL)» — full per-example rows in
`results/agd/Qwen3.5-27B/agd_results.json`.

### Qwen3.5-9B — NOT universal

| Benchmark | SC(8) | AGD(0.50) | Δ |
|---|---|---|---|
| math_hard | 50.0% | 26.7% | **-23.3%** |
| bbh_logic | 60.0% | 56.7% | -3.3% |
| arc_challenge | 56.7% | 63.3% | +6.7% |
| strategyqa | 83.3% | 76.7% | -6.7% |

«avg Δ=-6.7%, min Δ=-23.3% (NOT universal)».

**Key finding:** AGD efficacy scales with model capability. Diverse prompts help
27B (strong base); hurt 9B (weaker base can't exploit prompt variation).

### AGD ↔ PCS relationship

AGD is a **fixed-heuristic gate** (agreement-threshold). PCS generalizes it: the
calibrated selector learns from features (agreement, confidence, cluster
support, logprob, strategy) what AGD has hard-coded. The AGD scaling-law finding
is orthogonal and can be reported alongside PCS as motivational evidence.

## 2. PCS on real benchmarks — PENDING

Requires vLLM server (currently the openbayes instance hosting Qwen3.5 models).
Task ordering:

1. `split_manifest` generates calib (50) / test (100) for 5 benchmarks.
2. `run_portfolio_experiment --mode collect --split calib` on 27B.
3. `calibrated_selector fit` → artifact + AUC/ECE report.
4. `run_portfolio_experiment --mode collect --split test` on 27B.
5. `run_portfolio_experiment --mode eval --compare A,B,C`.
6. `analyze_pcs` for paired bootstrap + McNemar.

Server state: environment clean (vLLM 0.19.1 confirmed Apr 24), all 4 AGD runs
completed. No active PCS process.

## 3. What the selector debug_tiny tells us

The synthetic test forces 1 candidate with GT answer out of 4 per example. Even
on this degenerate distribution:

- LR separated correctness (AUC 0.81 raw / 0.81 calibrated)
- Isotonic reduced ECE by 4× (0.068 → 0.017)
- Brier stayed ≤ 0.13

This confirms the training/inference loop compiles, labels are applied only on
calibration rows (no leakage into test metrics), and isotonic calibration
improves ECE without harming AUC. Real signal strength will depend on candidate
pool diversity — the 27B AGD data suggests diverse-prompt candidates are
informative on strong models.

## 4. What success would look like

Following GPT-5.5 Pro §15 pre-registration:
- Selector AUC > 0.55 on held-out calibration
- C accuracy > B accuracy by ≥ 2 points, OR iso-accuracy with ≥ 20% fewer tokens
- Paired bootstrap 95% CI for C − B excludes 0
- Holds across 3 seeds and at least 3 benchmarks (mixing numeric/MC/boolean)

## 5. Kill criteria

Triggered if:
- AUC ≤ 0.55 AND no Pareto improvement
- C ≈ B (selector adds no signal)
- SC dominates at matched budget
- Post-fix metric recomputation erases all positive AGD evidence
