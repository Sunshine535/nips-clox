# Strict Metric Replay Results

> AUTO-GENERATED from `code/generate_strict_replay_report.py` over `results/agd/*/agd_results_strict.json`. Do not edit by hand. Run `python code/generate_strict_replay_report.py --check ...` as part of CI to verify markdown ↔ JSON consistency.

GPT-5.5 Pro Round 4 review identified that legacy `evaluation.check_answer` contained a substring fallback that contaminates accuracy on text/MC/boolean answers. This report shows what AGD numbers look like when re-scored through `check_answer_strict`.

Source script: `code/replay_results_strict.py` (committed).

## Qwen3.5-27B AGD

| Benchmark | Type | SC(8) legacy | SC(8) strict | Δ SC | AGD(0.5) legacy | AGD(0.5) strict | AGD-vs-SC strict |
|---|---|---:|---:|---:|---:|---:|---:|
| math_hard | math_expression | 0.400 | 0.300 | -0.100 | 0.567 | 0.500 | 0.200 |
| bbh_logic | multiple_choice | 0.300 | 0.033 | -0.267 | 0.067 | 0.000 | -0.033 |
| arc_challenge | multiple_choice | 0.833 | 0.333 | -0.500 | 0.867 | 0.400 | 0.067 |
| strategyqa | boolean | 0.833 | 0.800 | -0.033 | 0.800 | 0.767 | -0.033 |
| **mean** | | **0.592** | **0.367** | **-0.225** | **0.575** | **0.417** | **0.050** |

Source file: `results/agd/Qwen3.5-27B/agd_results_strict.json`.

## Qwen3.5-9B AGD

| Benchmark | Type | SC(8) legacy | SC(8) strict | Δ SC | AGD(0.5) legacy | AGD(0.5) strict | AGD-vs-SC strict |
|---|---|---:|---:|---:|---:|---:|---:|
| math_hard | math_expression | 0.233 | 0.200 | -0.033 | 0.267 | 0.200 | 0.000 |
| bbh_logic | multiple_choice | 0.567 | 0.033 | -0.533 | 0.567 | 0.033 | 0.000 |
| arc_challenge | multiple_choice | 0.633 | 0.067 | -0.567 | 0.633 | 0.033 | -0.033 |
| strategyqa | boolean | 0.800 | 0.633 | -0.167 | 0.767 | 0.700 | 0.067 |
| **mean** | | **0.558** | **0.233** | **-0.325** | **0.558** | **0.242** | **0.008** |

Source file: `results/agd/Qwen3.5-9B/agd_results_strict.json`.

## Findings

- **Qwen3.5-27B**:; math_hard +0.200; arc_challenge +0.067; bbh_logic -0.033; strategyqa -0.033
- **Qwen3.5-9B**:; strategyqa +0.067; math_hard ±0.000; bbh_logic ±0.000; arc_challenge -0.033

Strict replay weakens the legacy 'AGD universally non-negative on 27B' claim: under strict metric the headline is reduced to 'AGD non-negative on a subset of benchmarks; 9B remains capability-limited'.

## Reproduction

```bash
python code/replay_results_strict.py --input results/agd/Qwen3.5-27B/agd_results.json --out results/agd/Qwen3.5-27B/agd_results_strict.json
python code/replay_results_strict.py --input results/agd/Qwen3.5-9B/agd_results.json --out results/agd/Qwen3.5-9B/agd_results_strict.json
python code/generate_strict_replay_report.py --inputs results/agd/Qwen3.5-27B/agd_results_strict.json,results/agd/Qwen3.5-9B/agd_results_strict.json --out reports/STRICT_REPLAY_RESULTS.md
python code/generate_strict_replay_report.py --check reports/STRICT_REPLAY_RESULTS.md --inputs results/agd/Qwen3.5-27B/agd_results_strict.json,results/agd/Qwen3.5-9B/agd_results_strict.json
```
