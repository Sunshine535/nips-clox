# Next GPT-5.5 Pro Review Package

Bundle to send back to GPT-5.5 Pro for the next round of review.

## What to include in the bundle

1. **Commit range since last review:** `ded6f2b..HEAD` (this pass + priors)
2. **Reports:** every file under `reports/` including this one
3. **Test output:**
   ```
   python -m pytest tests/ -v  → 39 passed, 18 skipped
   python code/calibrated_selector.py debug_tiny  → AUC 0.81, ECE 0.017
   ```
4. **AGD final results:** `results/agd/Qwen3.5-{27B,9B}/agd_results.json`
5. **Current open questions (for GPT-5.5 Pro):**
   - Is it acceptable to report AGD 27B universal non-negative as a **secondary** finding in the PCS paper, or should it be a separate paper?
   - Is 50-example calibration sufficient, or should we budget for 100?
   - Should we invest in strict masked-slot repair now, or defer?
   - Is the "cross-strategy candidate-cluster calibration" framing sufficient to defend novelty vs Self-Certainty + RTR?

## Status summary

| GPT-5.5 Pro task | Status |
|---|---|
| 1 Freeze legacy path | ✓ `docs/legacy_status.md`; archived 7 files |
| 2 Immutable result schema | ✓ `code/result_schema.py` |
| 3 Strict answer extraction | ✓ `code/answer_extraction.py` + 32 tests |
| 4 Seed reproducibility | ✓ `set_global_seed`, `stable_hash_seed` + 7 tests |
| 5 Dataset split manifests | ✓ `code/split_manifest.py` (unpopulated until server run) |
| 6 Portfolio candidate generator | ✓ `code/portfolio.py` |
| 7 Calibrated selector | ✓ `code/features.py` + `code/calibrated_selector.py` |
| 8 Value-of-compute gate | ✓ `code/compute_gate.py` |
| 9 A/B/C evaluation | ✓ `code/run_portfolio_experiment.py` + `code/analyze_pcs.py` |
| 10 Rewrite paper thesis | Pending — blocked on PCS empirical run |

## What we're NOT claiming yet

- Any PCS > SC Pareto improvement on real benchmarks (no empirical run yet)
- Generalization across models (only 27B/9B for AGD)
- Superiority over Self-Certainty or FOBAR (no reproduction yet)

## Specific feedback we want

1. **Mechanism-level novelty check:** Does the candidate-cluster calibration
   formulation (cluster-level argmax with per-cluster isotonic-scored LR) differ
   meaningfully from published BoN / Self-Certainty / RTR? If not, what variant
   would?

2. **Minimum viable empirical claim:** Given the current AGD results, what is
   the most conservative PCS claim the experimental plan can falsify?
   - Option a) "PCS + AGD-style gate Pareto-dominates SC on ≥ 3 benchmarks on 27B"
   - Option b) "PCS converts oracle-SC gap to deployable gain when model is strong"
   - Option c) "Calibrated selector transfer across benchmarks with modest feature set"

3. **AGD replay priority:** Re-score the 27B AGD rows through
   `check_answer_strict` now (before any PCS run), or keep diagnostic status?

4. **Stop/pivot:** Given we have 27B universal AGD positives, is PCS still the
   right investment, or should we flip to writing "capability-gated diverse-prompt
   reasoning — a scaling-law empirical paper"?

## Code locations for review

- PCS logic flow: `code/portfolio.py` → `code/features.py` → `code/calibrated_selector.py` → `code/compute_gate.py` → `code/run_portfolio_experiment.py` → `code/analyze_pcs.py`
- Tests: `tests/test_answer_extraction.py`, `tests/test_seed_reproducibility.py`
- Reports: `reports/` (12 markdown files)

## Server context

- Host: openbayes job `od7riimbbymh` (ephemeral)
- Tunnel: `100.79.226.56:43039` → hpc.tju.edu.cn:31636
- Model path: `/openbayes/input/input0/hub/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd`
- vLLM: 0.19.1 (qwen3_5 arch OK)
- Hardware: 8 × GPUs, tp=4 for 27B

## Commit hashes

- `ded6f2b` AGD 27B+9B 4-benchmark final results
- `1d1cdd2` infra: legacy freeze, split manifests, result schema
- `76a7a91` P0: strict answer-type-aware extraction
- Next commit (post-this-pass): PCS pipeline complete + reports
