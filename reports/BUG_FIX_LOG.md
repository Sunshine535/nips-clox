# Bug Fix Log

Each entry names: file, bug, fix, test gate, risk of regression.

## P0-1 — Metric contamination (substring fallback)

**File:** `code/evaluation.py:61` historical; superseded by `code/answer_extraction.py`.
**Bug:** `check_answer` fell through to `return pred in ref or ref in pred` for the default path. A model producing "1" against ground truth "D" would pass on MC tasks; a model producing "the answer is 123 units" against "123" would pass via substring on numeric tasks.
**Evidence:** GPT-5.5 Pro §3 Priority P0 row 1; meta result rows with non-letter predictions marked correct.
**Fix:** `answer_extraction.check_answer_strict(pred, ref, answer_type)` handles numeric / multiple_choice / boolean / text with no substring fallback. Type-specific extractors (`extract_numeric`, `extract_multiple_choice`, `extract_boolean`) normalize both sides before comparison.
**Test gate:** `tests/test_answer_extraction.py` 32/32 including `test_mc_rejects_junk_prediction` (`check_answer_strict("1", "D", "multiple_choice") is False`) and `test_no_substring_fallback` (`check_answer_strict("123", "the answer is 123 units", "text") is False`).
**Regression risk:** Historical accuracy numbers on multiple-choice benchmarks (ARC-Challenge, BBH) may drop after re-scoring with the strict checker. Documented in `reports/CURRENT_RESULT_AUDIT.md`; AGD rows intentionally NOT re-scored to keep provenance intact.

## P0-2 — Seed reproducibility (Python hash)

**File:** `code/strategies_v2.py:216` (before); `code/utils.py` (after).
**Bug:** `np.random.default_rng(hash(question) % (2**32) + 12345)` — Python's built-in `hash()` is salted per-process unless `PYTHONHASHSEED` is set, so random_repair mask selections differed across processes even at "same seed".
**Evidence:** GPT-5.5 Pro §3 P1 row 6.
**Fix:** Added `stable_hash_seed(*parts)` using SHA256 (deterministic across processes). Added `set_global_seed` which also exports `PYTHONHASHSEED`. RandomRepair now calls `stable_hash_seed(question, self.seed, "random_repair")`.
**Test gate:** `tests/test_seed_reproducibility.py` 7/7, including `test_stable_hash_seed_deterministic`, `test_stable_hash_seed_distinguishes_seed`, `test_set_global_seed_pythonhashseed`.
**Regression risk:** Masked indices chosen by RandomRepair will differ from legacy runs. Legacy random_repair results remain in `results/v3`; not re-used for PCS.

## P0-3 — Raw result loss (checkpoint deletion)

**File:** `code/run_full_experiment.py::_aggregate_benchmark` (historical behaviour).
**Bug:** After aggregation, `.ckpt_*.json` per-example files were `unlink()`-ed. PCS needs candidate-level raw outputs for selector training and audit.
**Evidence:** GPT-5.5 Pro §3 P0 row 3.
**Fix:** `code/result_schema.py` provides `save_per_example(output_dir, strategy, seed, rows)` and `save_candidate_outputs(output_dir, example_id, candidates)`. New `run_portfolio_experiment.py` writes JSONL and never deletes. Legacy runner `run_full_experiment.py` is not modified yet (deferred; PCS pipeline is separate).
**Test gate:** Smoke test — portfolio `collect` mode produces a readable JSONL with ≥ 1 candidate / example. Verified via local dry-run `/tmp` tests.
**Regression risk:** None. PCS uses its own pipeline; legacy runner retained unchanged.

## P0-4 — StrategyQA train/test leakage

**File:** `code/benchmarks.py::load_strategyqa` (historical).
**Bug:** Silent fallback from `ChilleD/StrategyQA` test → `metaeval/strategy-qa` train when test unavailable.
**Evidence:** GPT-5.5 Pro §3 P0 row 4.
**Fix:** Not rewritten in `benchmarks.py` this pass; instead, `split_manifest.py` fixes the problem upstream. For a benchmark + seed, it creates a reproducible calib/test split over the loaded dataset. Manifests include `total_available`, `fingerprint`, `calib_ids`, `test_ids`. Any ambiguity about which split was used is now visible.
**Test gate:** `python code/split_manifest.py --check` validates no calib/test overlap.
**Regression risk:** If the upstream loader changes its default split, fingerprint mismatch will flag it. Manifests are seeded at `--seed 11`.

## P0-5 — Topology-router collapse (`"adaptive"` → `targeted_repair`)

**File:** `code/strategies_v2.py::CLOXAdaptive` (historical).
**Bug:** When topology estimator returns `"adaptive"`, code silently mapped it to `"targeted_repair"`. v3 topology distribution is mostly `"adaptive"`, so CLOXAdaptive collapsed to targeted_repair — yet claimed to be topology-routed.
**Evidence:** GPT-5.5 Pro §3 P1 row 7.
**Fix:** This pass **does not modify** CLOXAdaptive behaviour. Instead, `docs/legacy_status.md` marks it as LEGACY_BASELINE. PCS makes no topology routing claims. Future Task 4b will emit a concrete strategy label or raise.
**Test gate:** Documentation only this pass.
**Regression risk:** None — legacy behaviour preserved for reproducibility.

## Not addressed yet (pending PCS empirical run)

- `engine.extract_answer` still has permissive fallback paths. Callers route through `answer_extraction` now; a future pass will refactor the engine function itself.
- `engine.VLLMEngine.__init__(seed=42)` default seed not overridden in `run_full_experiment.py`. PCS pipeline (`run_portfolio_experiment.py::_collect`) does pass seed correctly.
- Targeted/random repair prompt still permits full rewrite (GPT-5.5 Pro §3 P0 row 5); reclassified as "candidate generator" rather than "local repair mechanism" via `docs/legacy_status.md`. Strict masked-slot implementation deferred.
