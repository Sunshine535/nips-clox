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
