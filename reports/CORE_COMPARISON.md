# Core Comparison — A vs B vs C

Per GPT-5.5 Pro §10 "Evaluation Protocol" and §15 "Minimal Verification".

## Definitions

| Arm | Meaning | Implementation |
|---|---|---|
| **A** | Existing Best Positive Fragment Only | `run_portfolio_experiment._old_fragment_pick` — majority vote among `sc_sample` candidates (SC-style, the strongest deployable non-selector baseline). Falls back to pool-wide majority vote if no SC samples. |
| **B** | Same portfolio, uncalibrated | `_majority_vote` — majority vote over ALL candidates (CoT + SC + repair + backward + full-regen). Tests whether adding strategy diversity alone (without calibrated selection) helps. |
| **C** | Full CLOX-PCS | `_pcs_pick` — calibrated selector scores every candidate, clusters by answer, picks argmax cluster. Value-of-compute gate records stop/continue decision. |

## Why these three, not others

- A is the strongest positive fragment currently. The argument "PCS is just a better SC" must be disproved by A, not something weaker.
- B isolates the contribution of **selector + calibration**. B has the same candidate pool as C but substitutes majority vote for calibrated scoring.
- C is the full method. Comparing C→B measures the selector's marginal value; comparing C→A measures the portfolio + selector value combined.

## Matched-budget protocol

All three arms read from the SAME `<split>_candidates.jsonl` file. Token cost
is accounted as the full pool (what was spent on collection). C's gate can
declare "stop" early at eval time, which would mean fewer tokens if we re-run
collection with the gate enabled; at A/B/C post-hoc comparison the budget is
the maximum (full pool) so A and B never get a token advantage over C.

## Statistical reporting

`code/analyze_pcs.py` emits:
- Per-arm accuracy, total tokens, mean tokens per example
- Pairwise `mean_diff_acc` with bootstrap CI (1000 resamples, seed=11)
- Pairwise McNemar p-value (binomial exact on discordant pairs)
- Per-pair `a_only_correct`, `b_only_correct`, `both_correct`, `both_wrong`

## Decision rule

Per GPT-5.5 Pro §15 success criterion:

> C beats B and compute-matched SC by ≥ 2 accuracy points OR saves ≥ 20%
> tokens at iso-accuracy across 3 seeds with paired CI excluding zero on at
> least the core benchmark set.

Core benchmark set: GSM8K, MATH-hard, StrategyQA, ARC-Challenge. Add BBH-logic
for mix-type robustness.

## Failure interpretation

| Observation | Interpretation | Action |
|---|---|---|
| C ≈ A | Old fragment good enough; selector adds no value | Stop PCS, pivot to AGD-style scaling-law paper |
| C ≈ B | Candidate diversity matters, calibration doesn't | Investigate: features too coarse? try logprob features |
| B > C | Bug: selector actively hurts | Debug selector; check leakage; check feature extraction |
| C > A, C > B, but CI includes 0 | Effect exists but underpowered | Expand N from 100 → 200, 3 seeds → 5 |
| All three ≈ SC | Answer pool doesn't contain oracle correct candidates | Report negative; examine oracle-any gap on this split |

## Not part of A/B/C (but required for paper)

Compute-matched SC (SC with token budget matched to PCS's actual consumption)
is reported alongside A/B/C as an external baseline. If PCS uses on average
5000 tokens/example and SC-at-5000-tokens-per-example achieves similar
accuracy, the case for PCS weakens significantly.

Self-Certainty and FOBAR reproductions are **pending**; GPT-5.5 Pro §16 lists
them as required official baselines before submission claims.
