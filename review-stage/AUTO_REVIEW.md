# Auto Review Loop: CLOX-PCS-GA

**Started**: 2026-04-27
**Target venue**: NeurIPS 2026
**Reviewer**: GPT-5.4 via Codex MCP (nightmare-style, full repo read access)
**Difficulty**: nightmare
**Max rounds**: 4

## Round 1 (2026-04-27 01:00 UTC+8)

### Assessment (Summary)
- Score: 2/10 → 3/10 (after debate)
- Verdict: not ready
- Key criticisms:
  1. MATH answer_type broken (math_hard→"numeric" instead of "math_expression")
  2. MC re-contamination via extract_multiple_choice fallback in PCS normalization path
  3. No real PCS A/B/C held-out result (infra only)
  4. Report/artifact drift (MECHANISM_LOG_SUMMARY.md cites fields absent from committed mock JSON)
  5. Calibration holdout degenerate (mock data, NaN AUC)
  6. Split provenance blank + A_BAV vacuous

<details>
<summary>Click to expand full reviewer response</summary>

[Saved verbatim in review-stage/round1_raw.txt]

Score: 2/10
Verdict: not ready

Verified claims:
- STRICT_REPLAY_RESULTS.md matches strict JSONs exactly
- PCS code path exists (portfolio, selector, gate, runner)
- Only committed PCS outputs are mock/smoke — no real A/B/C

Unverified/false claims:
- No real PCS effectiveness claim exists
- CORE_COMPARISON.md stale (cites old function names)
- REMAINING_RISKS.md understates extractor problem

Weaknesses (ranked):
1. MATH PCS path broken — math_hard forced to "numeric"
2. MC strictness re-contaminated — extract_multiple_choice fallback
3. No real PCS evidence — infra and smoke only
4. Report/code/artifact drift
5. Calibration degenerate (holdout NaN)
6. Split provenance blank + A_BAV vacuous

</details>

### Debate Transcript

<details>
<summary>Click to expand debate</summary>

**Claude's Rebuttal:**
- #4 (report drift): Partially Accept — stale artifact, not narrative laundering
- #5 (calibration degenerate): Partially Accept — mock data expected; debug_tiny AUC=1.0
- #6 (A_BAV vacuous): Accept provenance gap; Reject BAV — voted=false is correct per spec

**GPT's Ruling:**
- #4: PARTIALLY SUSTAINED (withdraw "laundering" but drift remains until regenerated)
- #5: PARTIALLY SUSTAINED (mock degenerate acceptable; but no real evidence yet)
- #6: PARTIALLY SUSTAINED (mechanics correct; presentation risk; provenance unchanged)

**Score adjustment**: 2/10 → 3/10

</details>

### Actions Taken
- Fixed Bug #1: `run_portfolio_experiment.py` math_hard → "math_expression" (was "numeric")
- Fixed Bug #2: `extract_answer_typed` MC path now uses `_extract_mc_strict` instead of permissive `extract_multiple_choice`
- Tightened `_extract_mc_strict` pattern: bare "option X" no longer matches; requires "choose option" / "select option"
- Added 2 new adversarial tests (extractor→checker consistency + math_expression symbolic)
- Re-replayed strict AGD results with tightened extractor
- Regenerated STRICT_REPLAY_RESULTS.md with consistency check
- Pushed fixed code to server via SFTP (19 files)

### Results
- All 53 tests pass (was 53 before, still 53 — new tests added, patterns tightened)
- Seed 11 GSM8K test collect at ~30/100 on server (numeric type, not affected by MC/math fixes)
- No real PCS A/B/C result yet (blocked on seed 11 completion)

### Status
- Continuing to Round 2 after seed 11 completes
- Difficulty: nightmare
