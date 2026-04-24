# Archived — CLOX v1 Legacy

Per GPT-5.5 Pro diagnosis (`GPT55_DIAGNOSIS.md` §13) these modules are **frozen
as historical evidence**. The new main method path is CLOX-PCS (Calibrated
Portfolio Compute Selection); see `docs/legacy_status.md` and
`reports/CLAUDE_EXECUTION_PLAN.md`.

## What's here

- `main.py`, `methods.py`, `strategies.py`, `topology.py` — v1 strategy suite
  (synthetic DAG era).
- `stage-13*.py` — stage-13 experiment scripts for the v1 pipeline.

## Why archived, not deleted

These files contain the historical code paths that produced the early
`EXPERIMENTS.md` results (targeted=random ablation collapse, etc.). Preserving
them keeps the audit trail intact for the final paper's "what failed and why"
section.

## Do not import from `code/`

None of these modules should be imported by the PCS pipeline. `code/` now
contains only files that participate in the new main path + their baselines.
