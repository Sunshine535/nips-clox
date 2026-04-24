# Current Result Audit

State-of-evidence as of commit `ded6f2b` (2026-04-24). Each row is a result
visible under `results/`, audited against the standard: (i) raw per-example
data present, (ii) manifest with command/seed/dataset/git-hash, (iii) metric
strict enough to trust for claims.

## Legend

- `raw`: per-example JSONL available
- `manifest`: manifest.json with command + seed + git_hash
- `metric`: answer check strict (numeric/MC/boolean enforced type)
- ✓ present · ✗ missing · ~ partial

## Results ledger

| Path | Scope | raw | manifest | metric | Usable for claim |
|---|---|:-:|:-:|:-:|---|
| `results/meta/` | 6×5 phase diagram (30 cells, N=30) | ~ | ✗ | ✗ (pre-fix) | Oracle-SC gap evidence only; do not report accuracy as method. |
| `results/pdsc/` | Prompt-diverse SC v1 | ✓ | ~ | ✗ | 3/4 positive; ARC negative; historical. |
| `results/pdsc_v2/` | PDSC v2 hybrid+confidence | ✓ | ~ | ✗ | Unused for paper; diagnostic. |
| `results/bav/` | Backward-anchored verification | ~ | ✗ | ✗ | agreed=disagreed=80%; agreement gate fails. Diagnostic. |
| `results/agd/Qwen3.5-27B` | AGD 4 bench N=30 | ✓ (json rows) | ✗ | ~ (legacy extractor) | **27B universally non-negative** (avg +4.2%, min +0%). |
| `results/agd/Qwen3.5-9B`  | AGD 4 bench N=30 | ✓ (json rows) | ✗ | ~ | 9B NOT universal (avg -6.7%, min -23.3%) — scaling law. |
| `results/pilot/` | Pilot 8-strategy | ~ | ✗ | ✗ | Historical, already diagnosed. |

## AGD — the only current positive result

**Status:** `results/agd/Qwen3.5-27B/agd_results.json` contains per-example rows
with `base_answers`, `div_answers`, `base_agreement`, `sc8_correct`,
`agd[thr].pred/correct/tokens/path`, ground truth.

**Reliability caveat:** AGD ran with the legacy `evaluation.check_answer` that
had the substring fallback. Confidence in the ±3% range is moderate; absolute
accuracy numbers are NOT directly comparable to post-fix PCS runs. To make AGD
commensurable with PCS:
- Replay AGD rows through `answer_extraction.check_answer_strict` (no rerun).
- If post-fix AGD remains universally non-negative on 27B, the scaling-law
  finding holds.

## Gaps that block PCS claims

- No calibration/test split manifests yet for any benchmark (Task 5 infra exists; `configs/splits/` empty).
- No per-candidate JSONL in any legacy result — PCS needs `portfolio collect` to start fresh.
- No selector artifact exists.

## What's claim-supportable NOW

1. Oracle-SC gap on 27B/bbh_logic up to 33% — diagnostic evidence.
2. AGD 27B 4-benchmark universal non-negativity, avg +4.2%, 27% token reduction on confidence-gated fast path.
3. AGD 9B 4-benchmark NOT universal (min -23.3% on math_hard) — capability-dependent finding.
4. BAV agreed=disagreed=80% — agreement gate useless, per-cluster calibration needed.
5. Topology r̄/ℓ correlate ≈ 0 with disagreement/diversity — hard rule fails.

## What's NOT claim-supportable yet

- No PCS over SC Pareto improvement — selector not yet fit.
- No multi-seed bootstrap CI for any of the above (N=30, single seed).
- No official baseline reproduction (Self-Certainty, FOBAR, RTR).
- No held-out test generalization — everything is test-on-train at current N.
