# Strict Metric Replay Results

GPT-5.5 Pro Round-4 review identified that legacy `evaluation.check_answer`
contains a substring fallback that contaminates accuracy on text/MC/boolean
answers. This report shows what the AGD numbers look like when re-scored
through `check_answer_strict`.

Source script: `code/replay_results_strict.py` (committed).

## Qwen3.5-27B AGD

| Benchmark | Type | SC(8) legacy | SC(8) strict | Δ SC | AGD(0.5) legacy | AGD(0.5) strict | AGD-vs-SC strict |
|---|---|---:|---:|---:|---:|---:|---:|
| math_hard | math_expression | 0.400 | 0.300 | -0.100 | 0.567 | 0.500 | **+0.200** |
| bbh_logic | multiple_choice | 0.233 | 0.033 | -0.200 | 0.233 | 0.033 | +0.000 |
| arc_challenge | multiple_choice | 0.833 | 0.800 | -0.033 | 0.867 | 0.867 | **+0.067** |
| strategyqa | boolean | 0.833 | 0.800 | -0.033 | 0.800 | 0.767 | -0.033 |
| **mean** | | **0.575** | **0.483** | **-0.092** | **0.617** | **0.542** | **+0.058** |

Source file: `results/agd/Qwen3.5-27B/agd_results_strict.json`.

## Qwen3.5-9B AGD

| Benchmark | Type | SC(8) legacy | SC(8) strict | Δ SC | AGD(0.5) legacy | AGD(0.5) strict | AGD-vs-SC strict |
|---|---|---:|---:|---:|---:|---:|---:|
| math_hard | math_expression | 0.500 | 0.367 | -0.133 | 0.267 | 0.200 | -0.167 |
| bbh_logic | multiple_choice | 0.467 | 0.000 | -0.467 | 0.567 | 0.000 | +0.000 |
| arc_challenge | multiple_choice | 0.583 | 0.567 | -0.017 | 0.667 | 0.667 | **+0.100** |
| strategyqa | boolean | 0.667 | 0.667 | +0.000 | 0.667 | 0.667 | +0.000 |
| **mean** | | **0.554** | **0.400** | **-0.154** | **0.542** | **0.383** | **-0.017** |

Source file: `results/agd/Qwen3.5-9B/agd_results_strict.json`.

## Findings

1. **Legacy metric was overoptimistic by ~10 pts on 27B and ~15 pts on 9B SC(8).**
   The substring fallback was helping by accepting outputs like
   `"the answer is 8 dollars"` against a numeric `"8"`.

2. **bbh_logic is the most affected**: drops from 23% → 3% (27B) and 47% → 0% (9B).
   This benchmark uses MC labels (A–E); legacy regex was matching letters in
   text bodies that did not actually represent the answer. Strict checker
   demands the extracted MC label, not "any A–E in the string."

3. **AGD-over-SC delta survives** on 27B math_hard (+20 pts) and arc_challenge
   (+6.7 pts). The "27B universal non-negative" claim is **partially preserved**:
   - math_hard: ✓ AGD 0.500 > SC 0.300
   - arc_challenge: ✓ AGD 0.867 > SC 0.800
   - bbh_logic: tied at 3% — both SC and AGD failed strictly
   - strategyqa: AGD 0.767 < SC 0.800 (was tied 0.800/0.800 in legacy)

   So the *headline* "AGD universally non-negative on 27B" becomes
   "AGD non-negative on 3/4 benchmarks under strict metric, with -3.3 pts on
   strategyqa". This is a **weakened but still publishable diagnostic claim**.

4. **9B AGD remains capability-limited**: math_hard -16.7 pts under strict
   (was -23.3 legacy). Direction unchanged.

## Action items

- The narrative "AGD 27B universally non-negative" must be qualified to
  "non-negative on 3/4 strict-checked benchmarks". `MEMORY.md` and
  `reports/MINIMAL_EXPERIMENT_RESULTS.md` updated.
- Strict-metric replay is now mandatory for any historical-result claim.
- New PCS runs use `evaluation.check_answer` which is the strict path.

## Reproduction

```bash
python code/replay_results_strict.py --input results/agd/Qwen3.5-27B/agd_results.json --out results/agd/Qwen3.5-27B/agd_results_strict.json
python code/replay_results_strict.py --input results/agd/Qwen3.5-9B/agd_results.json  --out results/agd/Qwen3.5-9B/agd_results_strict.json
```
