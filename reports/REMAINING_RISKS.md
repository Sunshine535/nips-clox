# Remaining Risks

## 1. Metric re-scoring on historical results

**Risk:** AGD, PDSC, meta, BAV results were computed under the legacy
`check_answer` with substring fallback. If we report them alongside PCS numbers
computed with `check_answer_strict`, we're comparing apples to oranges.

**Mitigation:** Replay per-example JSONL through `check_answer_strict`. Record
pre-fix vs post-fix accuracy side-by-side. If the delta is < 1 pt, the claim
survives; if not, retract to "diagnostic only".

**Severity:** high (could erase AGD scaling-law finding).

## 2. vLLM seed controllability

**Risk:** `VLLMEngine(seed=…)` sets the vLLM sampler seed, but vLLM tensor
parallelism with `tp>1` has nondeterministic reductions on some H100 stacks.
Two same-seed runs may differ in the 3rd-decimal accuracy.

**Mitigation:** Run 3 seeds (11, 23, 37) and report mean ± std. Paired bootstrap
CI uses per-example differences, which is robust to small across-run shifts.
Document "vLLM nondeterminism under TP" in the paper reproducibility section.

**Severity:** medium.

## 3. Calibration split size (50 examples)

**Risk:** 50 calibration examples × ~6 candidates = 300 training rows with ~40%
positive rate → ~120 positives. LR on 36-dim features should not overfit, but
feature importance estimates will be noisy.

**Mitigation:** Pre-register reporting to AUC / Brier / ECE only on held-out
calibration sub-split. Do not interpret individual feature weights. Scale to
N=100 calib if resources permit.

**Severity:** medium.

## 4. Oracle gap may disappear after metric fix

**Risk:** `results/meta/` oracle-SC gap was computed with legacy `check_answer`.
If strict checking recomputes oracle_any lower and SC roughly the same, the gap
could shrink from 33% → 10%. That would remove the "candidate pool has
selectable room" premise.

**Mitigation:** Re-score all meta cells. If gap > 10% on ≥ 3 cells, PCS is
still justifiable. Below that, pivot to AGD-style scaling-law paper or
negative-result paper.

**Severity:** high.

## 5. Novelty risk (Self-Certainty, RTR)

**Risk:** Both are very close in mechanism:
- Self-Certainty scores BoN candidates using output distribution confidence.
- Route-to-Reason routes across models/strategies under budget.

If we claim PCS "first calibrated cross-strategy answer-cluster selector", a
reviewer will cite Self-Certainty + RTR and reject.

**Mitigation:** Add Self-Certainty as both baseline AND feature (selector has
access to self-certainty scores). Reproduce RTR if code available. Frame the
contribution as "cross-strategy candidate-cluster calibration", not "first
adaptive selector".

**Severity:** very high.

## 6. Repair prompt full-rewrite confound

**Risk:** `UncertaintyTargetedRepair` and `RandomRepair` prompts permit full
solution rewrite, not just local slot-filling. Keeping them as candidate
generators means PCS may be benefiting from "rewrite diversity", not
"uncertainty targeting". A reviewer could argue this is just SC in disguise.

**Mitigation:** Implement strict masked-slot repair in a follow-up pass. Report
targeted_repair and random_repair SEPARATELY in PCS feature ablation; if the
selector performs same without their features, acknowledge.

**Severity:** medium.

## 7. Legacy extractor in engine.py

**Risk:** `engine.extract_answer` still has the old permissive fallback. Any
code still calling this (e.g., `strategies_v2.py` inside `StandardCoT`,
`SelfConsistency`, etc.) will produce candidates with permissively-extracted
normalized_answer. The PCS pipeline then calls `answer_extraction.extract_answer_typed`
on `raw_output`, which SHOULD override — but if there's a mismatch between the
engine's extracted prediction (logged into `candidate.extra["strategy_prediction"]`)
and the portfolio's re-extraction, the selector features may still reflect the
legacy extraction.

**Mitigation:** Audit `portfolio._to_candidate` — it calls
`extract_answer_typed(raw, answer_type)` which is the strict path. So
`normalized_answer` is strict. Legacy extraction only pollutes `extra.strategy_prediction`
which is not used by the selector. Low immediate risk.

**Severity:** low.

## 8. Paper thesis still unwritten

**Risk:** `paper/main.tex` currently claims the topology theorem as main. If
submitted in current state, the paper is factually inconsistent with the code.

**Mitigation:** Task 10 per GPT-5.5 Pro spec: rewrite thesis only after PCS
minimal tests pass. Until then, `docs/legacy_status.md` is the authoritative
statement. Paper rewrite is a follow-up task.

**Severity:** medium (no submission deadline yet).

## 9. Openbayes instance mortality

**Risk:** The server at `/input0/nips-clox` is an ephemeral rental. A restart
wipes local environment (as happened Apr 24) and requires vLLM re-install.

**Mitigation:** All commits pushed to GitHub. Environment install documented
via `setup.sh` and `code/requirements.txt`. PCS pipeline can be re-run from
scratch in ~30 min of setup + ~2 hr of collection at current scale.

**Severity:** medium (operational, not scientific).

## 10. Calibration / test set selection for AGD replay

**Risk:** AGD results are from 30 random per benchmark. They were NOT selected
via `split_manifest`. To replay through strict metric, we use the same 30;
but those 30 are now the "training set" for any subsequent claim, and claims
about generalization to other subsets are not supported.

**Mitigation:** Generate split manifests explicitly for PCS. Do NOT use AGD's
30-per-benchmark for PCS training. AGD remains its own diagnostic dataset.

**Severity:** low.
