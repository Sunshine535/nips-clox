# Local Repository Scan

## Snapshot
- Commit at time of scan: `ded6f2b data(AGD): complete 27B+9B × 4 benchmarks final results`
- Branch: `main`
- Working tree pre-PCS implementation (before this pass): 1 modified (9B AGD json), 1 untracked (GPT55_DIAGNOSIS.md), 1 untracked dir (27B AGD results)

## Active code files (`code/`)

| File | Role | Status |
|---|---|---|
| `engine.py` | vLLM wrapper, extract_answer | Active (targeted for Task 3 rewrite later) |
| `evaluation.py` | bootstrap/McNemar, check_answer | Active (substring fallback replaced by `answer_extraction.py`) |
| `benchmarks.py` | dataset loaders | Active |
| `strategies_v2.py` | v2 strategy suite | Active — RandomRepair updated to SHA256 seed (Task 4) |
| `topology_v2.py` | topology features | Active (will become proxy_features) |
| `run_clox.py`, `run_32b.py`, `run_full_experiment.py` | legacy runners | Kept for reproducibility |
| `answer_extraction.py` | strict type-aware extraction | NEW (Task 3) |
| `result_schema.py` | immutable manifest + JSONL | NEW (Task 2) |
| `split_manifest.py` | seeded calib/test splits | NEW (Task 5) |
| `utils.py` | seed helpers | EXTENDED (Task 4): `set_global_seed`, `stable_hash_seed` |
| `portfolio.py` | PCS candidate collection | NEW (Task 6) |
| `features.py` | PCS feature extractor | NEW (Task 7) |
| `calibrated_selector.py` | LR + isotonic | NEW (Task 7) |
| `compute_gate.py` | conservative stop-gate | NEW (Task 8) |
| `run_portfolio_experiment.py` | PCS collect/eval | NEW (Task 9) |
| `analyze_pcs.py` | paired bootstrap + McNemar | NEW (Task 9) |
| `agd.py`, `pdsc.py`, `pdsc_v2.py`, `meta_sweep.py`, `question_router.py`, `router_n150.py`, `analyze_disagreement.py`, `run_critical.py` | active experiment runners | Kept |

## Archived (moved this pass)

- `archive/legacy_clox_v1/main.py`
- `archive/legacy_clox_v1/methods.py`
- `archive/legacy_clox_v1/strategies.py`
- `archive/legacy_clox_v1/topology.py`
- `archive/legacy_clox_v1/stage-13_experiment.py`
- `archive/legacy_clox_v1/stage-13_v1_experiment.py`
- `archive/legacy_clox_v1/stage-13_v2_experiment.py`
- `archive/novelty_report.json`

## Results folder

- `results/meta/` — 6×5 phase diagram (KEEP)
- `results/pdsc/`, `results/pdsc_v2/` — ablation history (KEEP)
- `results/bav/` — BAV diagnostic (KEEP)
- `results/agd/Qwen3.5-27B` and `Qwen3.5-9B` — 4-benchmark complete (LATEST)
- `results/pilot/` — historical (KEEP)

## Tests

| File | Scope | Result |
|---|---|---|
| `tests/test_answer_extraction.py` | 32 strict extraction tests | 32 pass |
| `tests/test_seed_reproducibility.py` | 7 seed/hash tests | 7 pass |
| `tests/test_clox.py` | legacy v1 unit tests | 18 skipped (archived) |

## Docs

- `docs/legacy_status.md` — freezes CLOXAdaptive as LEGACY_BASELINE
- `reports/CLAUDE_EXECUTION_PLAN.md` — GPT-5.5 Pro execution manifest
- `MEMORY.md` — cross-conversation notes (gitignored, in `~/.claude/.../memory/`)

## Out of scope / frozen

- Paper rewrite (`paper/main.tex`) — Task 10 per spec; defer until PCS empirical.
- Full-benchmark PCS run — pending live vLLM on server.
