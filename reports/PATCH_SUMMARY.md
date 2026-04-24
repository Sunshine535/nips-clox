# Patch Summary

Diff-level summary of every code change introduced by this GPT-5.5 Pro execution
pass. Each entry shows file, lines, intent, and the task it maps to.

## NEW files

| File | Lines | Task | Purpose |
|---|---:|:---:|---|
| `code/answer_extraction.py` | ~140 | 3 | Strict type-aware extract + check; no substring fallback. |
| `code/result_schema.py` | ~75 | 2 | `create_run_manifest`, `save_per_example`, `save_candidate_outputs`. |
| `code/split_manifest.py` | ~110 | 5 | Seeded calib/test fixed-ID manifests. |
| `code/portfolio.py` | ~170 | 6 | `run_portfolio`, `cluster_candidates`, `summarize_clusters`. |
| `code/features.py` | ~105 | 7 | `FEATURE_NAMES`, `extract_features`, `build_feature_matrix`, `build_labels`. |
| `code/calibrated_selector.py` | ~220 | 7 | LR + isotonic fit / score / save / load; `fit`, `report`, `debug_tiny` commands. |
| `code/compute_gate.py` | ~90 | 8 | `decide`, `pick_best_answer`, `aggregate_cluster_scores`. |
| `code/run_portfolio_experiment.py` | ~215 | 9 | `collect` and `eval` modes with A/B/C comparison. |
| `code/analyze_pcs.py` | ~75 | 9 | Paired bootstrap + McNemar post-hoc analysis. |
| `tests/test_answer_extraction.py` | ~140 | 3 | 32 tests; strict / no substring / MC-junk / boolean / numeric. |
| `tests/test_seed_reproducibility.py` | ~55 | 4 | 7 tests; SHA256 stability, same-seed determinism, PYTHONHASHSEED. |
| `docs/legacy_status.md` | ~40 | 1 | Freezes CLOXAdaptive as LEGACY_BASELINE. |
| `archive/legacy_clox_v1/README.md` | ~25 | 1 | Explains archival scope. |
| `reports/CLAUDE_EXECUTION_PLAN.md` | ~75 | — | Per spec: MAIN METHOD, evidence, files to edit/archive. |
| `reports/LOCAL_REPO_SCAN.md` | (this set) | — | Active vs archived layout. |
| `reports/GPT55_REPORT_EXTRACTION.md` | (this set) | — | Distilled execution path. |
| `reports/CURRENT_RESULT_AUDIT.md` | (this set) | — | Claim-supportable vs not. |
| `reports/KEEP_REWRITE_ARCHIVE_PLAN.md` | (this set) | — | Per §13 decision. |
| `reports/BUG_FIX_LOG.md` | (this set) | — | Per-fix test gates. |
| `reports/TEST_PLAN.md` | (this set) | — | Gates run + gates planned. |
| `reports/MINIMAL_EXPERIMENT_RESULTS.md` | (this set) | — | AGD evidence + PCS pending. |
| `reports/CORE_COMPARISON.md` | (this set) | — | A/B/C definitions and decision rule. |
| `reports/CLAIM_UPDATE_LOG.md` | (this set) | — | Retained / weakened / retracted. |
| `reports/REMAINING_RISKS.md` | (this set) | — | Known unresolved risks. |
| `reports/NEXT_GPT55_REVIEW_PACKAGE.md` | (this set) | — | What to send next to GPT-5.5 Pro. |

## MODIFIED files

| File | Change | Task |
|---|---|:---:|
| `code/utils.py` | Added `set_global_seed` (alias + PYTHONHASHSEED), `stable_hash_seed` (SHA256). `stable_int_seed` now delegates to SHA256 hash instead of MD5. | 4 |
| `code/strategies_v2.py` | `RandomRepair.__init__` gains `seed: int = 11`. Line 216 replaces `hash(question)` with `stable_hash_seed(question, self.seed, "random_repair")`. | 4 |
| `tests/test_clox.py` | `pytestmark = pytest.mark.skip(...)`; import path routed to `archive/legacy_clox_v1/`. 18 tests skip. | 1 |
| `results/agd/Qwen3.5-9B/agd_results.json` | Fresh 4-benchmark results (new strategyqa rows added). | — (AGD run, prior commit) |

## MOVED via `git mv`

| From | To |
|---|---|
| `code/main.py` | `archive/legacy_clox_v1/main.py` |
| `code/methods.py` | `archive/legacy_clox_v1/methods.py` |
| `code/strategies.py` | `archive/legacy_clox_v1/strategies.py` |
| `code/topology.py` | `archive/legacy_clox_v1/topology.py` |
| `code/stage-13_experiment.py` | `archive/legacy_clox_v1/stage-13_experiment.py` |
| `code/stage-13_v1_experiment.py` | `archive/legacy_clox_v1/stage-13_v1_experiment.py` |
| `code/stage-13_v2_experiment.py` | `archive/legacy_clox_v1/stage-13_v2_experiment.py` |
| `docs/novelty_report.json` | `archive/novelty_report.json` |

## Prior passes (already on main)

- `76a7a91` fix(P0): strict answer-type-aware extraction, remove substring fallback
- `1d1cdd2` infra(P0): legacy freeze, split manifests, result schema
- `ded6f2b` data(AGD): complete 27B+9B × 4 benchmarks final results

## Total delta (this pass only)

- Code: +1,065 lines added (new), ~30 lines modified
- Tests: +195 lines added (2 new files)
- Reports: +12 markdown docs
- Archive: 7 files moved + 1 README

## Verification

```
python -m compileall code           # all clean
python -m pytest tests/             # 39 passed, 18 skipped
python code/split_manifest.py --check   # no manifests yet, all OK
python code/calibrated_selector.py debug_tiny   # AUC 0.81, ECE 0.068→0.017
```
