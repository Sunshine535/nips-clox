# Claude Code Execution Plan

## 1. GPT55_DIAGNOSIS.md Location
`/home/tarkoy/nips/nips-clox/GPT55_DIAGNOSIS.md` (root, 1202 lines)

## 2. MAIN METHOD PATH
**CLOX-PCS: Calibrated Portfolio Compute Selection**
- Demote all existing strategies to candidate generators
- Add calibrated selector trained on held-out calibration split
- Add value-of-compute gate for adaptive budget control
- Core novelty: cross-strategy answer-cluster calibration, not generic routing

## 3. Missing Mechanism
**Outcome-aware calibrated selection + compute gating**: predict which candidate answer is correct and whether more compute is worth spending.

## 4. Current Evidence Supporting Diagnosis
- Oracle-SC gap exists (max 43.3% on 27B/bbh_logic) — candidate pool has room
- PDSC v1/v2 show diversity helps conditionally — selection needed
- All routers/selectors failed (2-16% gap capture) — input features insufficient
- BAV gate non-discriminative (agreed=disagreed=80%) — crude gate fails
- AGD experiment shows diversity hurts 9B but helps 27B — model-dependent

## 5. Current Evidence Contradicting/Weakening Diagnosis
- AGD 9B math_hard: SC=50%, AGD=27% — diverse candidates very weak on 9B
- PDSC v2 shows no universally positive config — selection may face same issue
- N=30 per cell gives high variance — effects may be noise
- Reviewer identified DDPrompt/DIPPER/DiVeRSe as close prior art

## 6. Files to Inspect
- `code/evaluation.py` — metric contamination risk (P0)
- `code/engine.py` — extract_answer fallback (P0)
- `code/benchmarks.py` — StrategyQA train fallback (P0)
- `code/strategies_v2.py` — repair prompt confounds, CLOXAdaptive (P0)
- `code/run_full_experiment.py` — seed/checkpoint/aggregation (P0)
- `code/topology_v2.py` — proxy validity (P1)

## 7. Files to Edit
- `code/evaluation.py` — rewrite check_answer by type
- `code/engine.py` — refactor extract_answer
- `code/benchmarks.py` — add split manifest, fix StrategyQA
- NEW: `code/answer_extraction.py`
- NEW: `code/result_schema.py`
- NEW: `code/split_manifest.py`
- NEW: `code/portfolio.py`
- NEW: `code/features.py`
- NEW: `code/calibrated_selector.py`
- NEW: `code/compute_gate.py`
- NEW: `code/run_portfolio_experiment.py`
- NEW: `tests/test_answer_extraction.py`
- NEW: `tests/test_metrics.py`

## 8. Files to Archive
- `code/main.py`, `code/methods.py`, `code/strategies.py`, `code/topology.py` → `archive/legacy_clox_v1/`
- `code/stage-13*.py` → `archive/legacy_clox_v1/`
- `docs/novelty_report.json` → `archive/`

## 9. Files NOT to Touch
- `results/meta/` — historical phase diagram data (keep as evidence)
- `results/pilot/` — historical pilot data
- `results/bav/` — keep as baseline evidence
- `results/pdsc/`, `results/pdsc_v2/` — keep as ablation evidence
- `EXPERIMENTS.md` — historical negative evidence
- `RESEARCH_BRIEF.md` — lessons learned

## 10. Tests Before/After
- Before: `python -m compileall code` (baseline compile check)
- After Task 3: `pytest tests/test_answer_extraction.py tests/test_metrics.py`
- After Task 5: `python code/split_manifest.py --check`
- After Task 6-8: smoke test with 2 examples
- After Task 9: A/B/C comparison on minimal set

## 11. Rollback Conditions
- Any compile failure → revert last change
- Metric fix erases ALL positive signal → document, consult
- Selector AUC ≤ 0.55 → report as negative, do not cherry-pick
- SC dominates at matched budget across all tasks → stop PCS, write negative report
