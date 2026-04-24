# Keep / Rewrite / Archive Plan

Executed per GPT-5.5 Pro §13. "Done" = action taken in this pass; "Pending" = future
phase.

## Kept (as-is, minimum touch)

| Path | Why |
|---|---|
| `code/engine.py` | vLLM wrapper works; only `extract_answer` will be refactored out later. |
| `code/benchmarks.py` | Loaders still in use; split logic handled externally in `split_manifest.py`. |
| `code/evaluation.py` | Stats code kept; answer-check delegated to `answer_extraction.py`. |
| `code/strategies_v2.py` | Strategies serve as candidate generators now. Only `RandomRepair` modified. |
| `code/topology_v2.py` | Will be demoted to proxy-features module in a later pass; kept functional. |
| `results/meta/`, `results/pilot/`, `results/bav/`, `results/pdsc/`, `results/pdsc_v2/` | Historical evidence for paper appendix. |
| `EXPERIMENTS.md`, `RESEARCH_BRIEF.md` | Failure audit trail — GPT-5.5 Pro explicitly says keep. |

## Rewritten this pass

| Path | Change | Status |
|---|---|---|
| `code/utils.py` | Added `set_global_seed`, `stable_hash_seed` (SHA256), `PYTHONHASHSEED` export | Done |
| `code/strategies_v2.py::RandomRepair` | Replaced `hash(question)` with `stable_hash_seed`; added `seed` kwarg | Done |
| `tests/test_clox.py` | Skip-marked; import path routed to `archive/legacy_clox_v1/` | Done |

## Added this pass

| Path | Role |
|---|---|
| `code/answer_extraction.py` | strict type-aware extractor + checker |
| `code/result_schema.py` | immutable manifest + per-example JSONL |
| `code/split_manifest.py` | seeded calib/test splits |
| `code/portfolio.py` | candidate pool generator |
| `code/features.py` | fixed-size per-candidate features |
| `code/calibrated_selector.py` | LR + isotonic + AUC/Brier/ECE |
| `code/compute_gate.py` | conservative stop gate |
| `code/run_portfolio_experiment.py` | A/B/C evaluation runner |
| `code/analyze_pcs.py` | paired bootstrap + McNemar + pair-diff tables |
| `tests/test_answer_extraction.py` | 32 regression tests |
| `tests/test_seed_reproducibility.py` | 7 seed/hash tests |
| `docs/legacy_status.md` | freezes CLOXAdaptive |
| `reports/CLAUDE_EXECUTION_PLAN.md` | execution manifest |
| `reports/*.md` (this set) | audit + status |
| `archive/legacy_clox_v1/` | moved v1 code, preserved git history via `git mv` |

## Archived (not deleted)

| Path → | Why kept |
|---|---|
| `archive/legacy_clox_v1/main.py` | synthetic-DAG era entry point |
| `archive/legacy_clox_v1/methods.py` | v1 method wrappers |
| `archive/legacy_clox_v1/strategies.py` | v1 strategy suite — targeted=random ablation |
| `archive/legacy_clox_v1/topology.py` | v1 topology estimator |
| `archive/legacy_clox_v1/stage-13_experiment.py` | stage-13 v1 runner |
| `archive/legacy_clox_v1/stage-13_v1_experiment.py` | stage-13 v1 runner |
| `archive/legacy_clox_v1/stage-13_v2_experiment.py` | stage-13 v1 runner |
| `archive/novelty_report.json` | stale (missed DDPrompt/DIPPER); GPT-5.5 Pro said ARCHIVE |
| `archive/legacy_clox_v1/README.md` | explains archival reason (this pass) |

## Pending (deferred until PCS empirical data exists)

| Path | Planned action |
|---|---|
| `code/engine.py::extract_answer` | refactor to call `answer_extraction.extract_answer_typed` |
| `code/evaluation.py::check_answer` | delegate to `answer_extraction.check_answer_strict` |
| `code/topology_v2.py` | rename → `topology_proxy_features.py` once PCS uses it as features only |
| `code/run_full_experiment.py` | integrate `result_schema.create_run_manifest` / `save_per_example` |
| `paper/main.tex` | rewrite thesis to PCS after empirical A/B/C test passes |
| `README.md` | update progress table, add PCS section, cite AGD scaling law |

## Do-not-touch list

- Any `.jsonl` or `.json` under `results/meta/`, `results/pilot/`, `results/bav/`, `results/pdsc/`, `results/pdsc_v2/`, `results/agd/`.
- `EXPERIMENTS.md`, `RESEARCH_BRIEF.md`, `IDEA_REPORT.md` — historical audit trail.
- Any external server-side log — pull-only via SFTP.
