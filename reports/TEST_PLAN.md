# Test Plan

## Existing tests (all pass or skip)

```
tests/test_answer_extraction.py   32 passed    — strict extraction / type-aware check
tests/test_seed_reproducibility.py 7 passed    — SHA256 hash + set_global_seed
tests/test_clox.py                18 skipped  — legacy v1 strategies (archived)
```

## Sanity checks run this pass

| Command | Purpose | Result |
|---|---|---|
| `python -m compileall code` | syntax validity across all active modules | all compile |
| `python code/split_manifest.py --check --output configs/splits` | cross-validate calib/test no overlap | empty manifests → "All manifests valid." |
| `python code/calibrated_selector.py debug_tiny` | selector self-overfit on toy | AUC 0.81, ECE 0.068→0.017 after isotonic |
| smoke: `portfolio.cluster_candidates` | cluster-id assignment stable | cross-strategy answers "42"/"42"/"40" → clusters {0,0,1} |
| smoke: `compute_gate.decide` | stop/continue/budget-exhaust paths | 3/3 paths correct |
| smoke: `run_portfolio_experiment._eval + analyze_pcs.analyze` | end-to-end A/B/C on synthetic rows | paired bootstrap CI + McNemar both reported |

## Planned tests (pre-empirical PCS run)

These are the "minimal verification" tasks GPT-5.5 Pro §15 requires before scale-up:

1. **One-batch overfit (Task 7 verification):**
   `python code/calibrated_selector.py debug_tiny` → AUC >0.55 on toy. Currently passes.

2. **Data sanity** (needs benchmarks loaded):
   `python code/split_manifest.py --benchmarks gsm8k,math_hard,strategyqa,arc_challenge,bbh_logic --n_calib 50 --n_test 100 --seed 11`
   Expected: five manifest JSONs under `configs/splits/`, all with `total_available` > 150.

3. **Portfolio collect smoke** (needs vLLM + model):
   `python code/run_portfolio_experiment.py --mode collect --benchmark gsm8k --max_examples 2 --model <model> --out results/pcs/smoke/cands.jsonl`
   Expected: ≥ 2 rows, each with ≥ 3 candidates (sc_sample + standard_cot + others).

4. **Selector fit on real data** (needs collect output):
   `python code/calibrated_selector.py fit --calib results/pcs/minimal/calib.jsonl --out results/pcs/minimal/selector.pkl`
   Acceptance: AUC > 0.55, ECE < 0.20.

5. **A/B/C eval** (needs selector + test candidates):
   `python code/run_portfolio_experiment.py --mode eval --compare A,B,C --candidates ... --selector ... --out eval.json`
   Acceptance per GPT-5.5 Pro §15: C > B by ≥ 2 accuracy points OR iso-accuracy token reduction ≥ 20%, with paired bootstrap CI excluding 0 across 3 seeds.

## Regression tests that must stay green

- Substring fallback forbidden: `check_answer_strict("1", "D", "multiple_choice") is False`
- Same-seed same-output: `stable_hash_seed("q", 11) == stable_hash_seed("q", 11)`
- Cross-process determinism requires `set_global_seed` before any strategy with stochastic logic runs.

## Test budget

Each pytest invocation < 3s total. Adding new tests per module should keep total suite under 10s.
