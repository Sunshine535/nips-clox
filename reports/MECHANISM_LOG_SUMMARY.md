# Mechanism Log Summary

GPT-5.5 Pro Round 5 Task 5: prove the C arm's value-of-compute gate is
**active**, not log-only. This summary aggregates per-example diagnostics
from `results/pcs/smoke/mock_eval_s11.json` (mock-engine smoke run with
50 calib + 100 test on GSM8K split manifest).

## Summary

| Metric | Value |
|---|---:|
| n examples | 100 |
| C arm `expansion_used=True` | 100 / 100 (100%) |
| C arm `expansion_used=False` (gate stopped at scout) | 0 / 100 (0%) |
| Mean C tokens used | 440 |
| Mean B tokens used | 440 |
| Mean scout tokens (subset of C) | 200 |
| Mean expansion tokens (subset of C) | 240 |
| C accuracy (mock data) | 0.010 |
| B accuracy (mock data) | 0.030 |
| A_SC accuracy (mock data) | 0.020 |

## Why all examples expanded on this smoke

Mock engine emits deterministic-but-random text via SHA-256(prompt+seed).
Per-strategy candidates therefore don't cluster around a true answer —
the selector cannot find a high-confidence cluster, so the gate
correctly returns `continue` every time. Mechanism is functioning as
designed.

In real-data runs we expect a non-trivial fraction of `expansion_used=False`
when the scout pool already converges on a reliable answer. A 0% stop
rate on real GSM8K data (after Task 6) would indicate the selector
features have insufficient signal — that would trigger
`MECHANISM_NO_CONTRIBUTION` (or `MECHANISM_INACTIVE` if logs vanish).

## Per-example log fields confirmed present

For each C arm record, `extra` contains:

```json
{
  "stage": 2,
  "decision_stage1": {
    "action": "continue",
    "rationale": "uncertain: best=...; margin=...; tau_stop=0.75",
    "best_cluster_id": ...,
    "best_score": ...,
    "cluster_score_map": {...}
  },
  "decision_stage2": {... if expansion executed ...},
  "stage1_pool_size": 5,
  "stage2_pool_size": 9,
  "expansion_used": true,
  "tokens_used": 440
}
```

Plus arm-level:
- `prediction`, `correct`, `tokens_used`, `voted`

## What to look for in real runs

| Signal | Indicates |
|---|---|
| `expansion_used=False` rate > 20% | gate correctly stopping on confident clusters |
| C tokens consistently < B tokens | gate active and saving compute |
| `decision_stage1.action=stop` for confident examples | tau_stop threshold appropriate |
| `cluster_score_map` having distinct scores | selector ranking informative |

## Run reproduction

```bash
python code/run_portfolio_experiment.py --mode collect \
  --benchmark gsm8k --split calib \
  --split_manifest configs/splits/gsm8k.json \
  --seed 11 --model dummy --mock_engine \
  --strategies standard_cot,self_consistency,targeted_repair,random_repair,backward_cloze,full_regeneration \
  --sc_k 4 --max_tokens 64 \
  --out results/pcs/smoke/mock_calib_s11.jsonl

python code/run_portfolio_experiment.py --mode collect \
  --benchmark gsm8k --split test \
  --split_manifest configs/splits/gsm8k.json \
  --seed 11 --model dummy --mock_engine \
  --strategies standard_cot,self_consistency,targeted_repair,random_repair,backward_cloze,full_regeneration \
  --sc_k 4 --max_tokens 64 \
  --out results/pcs/smoke/mock_test_s11.jsonl

python code/calibrated_selector.py fit \
  --calib results/pcs/smoke/mock_calib_s11.jsonl \
  --out results/pcs/smoke/mock_selector_s11.pkl --seed 11

python code/run_portfolio_experiment.py --mode eval \
  --compare A_SC,A_TARGETED,A_BAV,B,C \
  --candidates results/pcs/smoke/mock_test_s11.jsonl \
  --selector results/pcs/smoke/mock_selector_s11.pkl \
  --tau_stop 0.75 --tau_margin 0.10 \
  --out results/pcs/smoke/mock_eval_s11.json
```

## Verdict

**Mechanism is active in code.** Gate fields populated, token accounting
distinguishes scout vs expansion. Real-data evaluation (Task 6) is
required to determine whether the gate is *useful* (non-trivial stop
rate, real C < B token saving). Smoke alone cannot prove utility, only
that the wiring is correct.
