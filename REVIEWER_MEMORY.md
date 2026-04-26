# Reviewer Memory

Persistent across review rounds. GPT-5.4 (via codex exec) uses this to track suspicions, patterns, and unresolved concerns.

Nightmare mode: GPT has full read access to the repo; Claude cannot curate.

## Round 1 — Score: 3/10 (not ready)

### Verified claims
- BAV implementation correct (1 forward + 1 backward, fallback k=5 on disagreement)
- Token accounting internally consistent
- Paired bootstrap numbers match the code

### Suspicions (unresolved)
- **MoT miscitation**: arXiv:2310.03094 is a cascade paper, NOT "pools CoT+PoT". Author's novelty framing built on wrong citation.
- **backward_cloze buried**: BAV 40/50 vs backward_cloze 39/50, p≈0.66. BAV's "positive result" is essentially backward_cloze alone.
- **FOBAR is closer prior art** than draft admits.
- **Token bimodality hidden**: agreement path 4.1k vs disagreement 15.5k (not "~13k" as claimed).
- **Agreement signal is decorative**: the gain comes from SC(k=5) fallback's extra compute on disagreements, not from agreement as a confidence filter.

### Patterns
- Author is honest about N=50 underpower and agreement-signal failure (rare strength)
- But author SYSTEMATICALLY downplayed the strongest baseline (backward_cloze) and inflated novelty via miscitation
- Compute story is "prettier than reality"

### Unresolved for future rounds
- Whether Author will correct 2310.03094 miscitation or inflate further
- Whether backward_cloze is restored to the main comparison
- Whether matched adaptive baselines are added (BAV+SC3 fallback, SC3→SC5 escalation, backward_cloze→SC)

## Termination (before Round 2)

Author chose to TERMINATE the loop at Round 1 rather than continue. Honest reason: pilot data does not support publishable positive result. BAV ≈ backward_cloze in accuracy (p=0.66), and all other concerns (citation, matched baselines, scale) would not close the gap without finding a genuinely new signal.

This terminates the loop at max_rounds=1/4 with status="terminated_honestly" rather than "ready".

---

## New Direction: PDSC (Prompt-Diverse Self-Consistency) — Pre-Round 2

### Context since Round 1
After BAV termination, the author:
1. Completed a 6×5 meta-sweep (6 models × 5 benchmarks, N=30 each): systematic Oracle-SC gap characterization
2. Ran N=150 validation on 3 high-gap cells: confirmed gap stability
3. Tried 3 different selector/router approaches: all failed (2-16% gap capture)
4. Root cause: Oracle-SC gap comes from stochastic strategy complementarity, not feature-predictable patterns

### New hypothesis (PDSC)
Standard SC uses k=8 samples from ONE prompt → temperature-only diversity → correlated errors.
PDSC distributes k=8 across 8 different reasoning prompts → strategy diversity → less correlated errors → higher majority-vote accuracy.

Same compute budget (k=8 total samples), but with prompt diversity instead of temperature-only diversity.

### Concerns to verify
- Is PDSC genuinely novel vs Universal Self-Consistency (Chen 2023)?
- Are the 8 prompts truly diverse or cosmetically different?
- Does prompt diversity actually reduce error correlation?
- Is the phase diagram finding (where gaps are large) sufficient novelty by itself?

## Major Pivot: CLOX-PCS-GA (post-GPT-5.5 Pro Diagnosis)

### Context since PDSC
1. GPT-5.5 Pro (1201-line audit) diagnosed: topology theorem unsupported, metric contaminated, AGD "universal" claim inflated, PCS needed calibrated selector + active compute gate
2. Claude Code implemented PCS skeleton: portfolio.py, features.py, calibrated_selector.py, compute_gate.py, run_portfolio_experiment.py, analyze_pcs.py
3. Round 4 GPT review found 7 blocking bugs (model_name, substring fallback, etc.) — all fixed
4. Round 5 GPT review found: MC checker too permissive, reports inconsistent, gate was inactive — fixed
5. Strict metric replay showed legacy metric inflated accuracy by ~10 pts on 27B
6. MC tightening dropped ARC/BBH by ~30+ pts — legacy results were heavily contaminated
7. PCS seed 11 GSM8K experiment now running (calib done, test ~30/100)
8. No real PCS A/B/C result yet

### Unresolved suspicions to carry forward
- Does PCS selector actually add value over majority vote (C>B)?
- Is the "active staged gate" actually saving tokens vs full-pool?
- Novelty risk vs Self-Certainty/RTR remains VERY HIGH
- AGD scaling-law finding is weakened (only math_hard strongly positive under strict metric)
- The 27B model generates max_tokens=4096 even for GSM8K — inference speed is ~3 ex/hr on tp=8 which is very slow

## Round 1 (Fresh Loop) — Score: 2/10 (not ready)

### Verified by GPT
- STRICT_REPLAY_RESULTS.md matches strict JSONs exactly
- PCS code path exists (portfolio, selector, gate, runner)
- Only committed PCS outputs are mock/smoke — no real A/B/C result

### New suspicions
- **MATH answer_type broken**: runner forces math_hard→"numeric" but loader emits "math_expression"; symbolic answers mangled
- **MC re-contamination**: extract_multiple_choice() fallback grabs any A-E in last 3 lines → feeds clean letter to strict checker → bypasses tightening
- **Report/code drift**: MECHANISM_LOG_SUMMARY.md claims fields absent from actual mock_eval JSON
- **Calibration degenerate**: holdout_positive_rate=0.0, holdout_auc=NaN in mock selector metrics
- **Split provenance blank**: gsm8k.json source_repo="" 
- **A_BAV vacuous**: default eval includes A_BAV but default collect omits bav strategy

### Patterns
- Author keeps finding and fixing bugs reactively; each round reveals more contamination vectors
- Report generation is not automated end-to-end — markdown drifts from artifacts
- The "strict metric" fix is not fully propagated through the new PCS extractor path
