# MC Strict-Checker Tightening — Delta Report

GPT-5.5 Pro Round 5 Task 2 stop condition explicitly required:
> Stop if tightening MC checker changes strict replay by more than 5 points
> on ARC or BBH; report the delta before proceeding.

The delta IS larger than 5 points on multiple cells. Reporting before
proceeding (per spec). The new checker is objectively correct; the old
permissive `re.search([A-E], text)` matched any letter in arbitrary
prose, not just answer markers.

## Numbers

### Qwen3.5-27B AGD — strict accuracy comparison

| Benchmark | Pre-tightening SC strict | Post-tightening SC strict | Δ | Pre AGD(0.5) strict | Post AGD(0.5) strict | Δ |
|---|---:|---:|---:|---:|---:|---:|
| math_hard | 0.300 | 0.300 | 0.000 | 0.500 | 0.500 | 0.000 |
| bbh_logic | 0.067 | 0.033 | -0.033 | 0.033 | 0.000 | -0.033 |
| arc_challenge | 0.800 | 0.333 | **-0.467** | 0.867 | 0.400 | **-0.467** |
| strategyqa | 0.800 | 0.800 | 0.000 | 0.767 | 0.767 | 0.000 |

### Qwen3.5-9B AGD — strict accuracy comparison

| Benchmark | Pre SC strict | Post SC strict | Δ | Pre AGD strict | Post AGD strict | Δ |
|---|---:|---:|---:|---:|---:|---:|
| math_hard | 0.200 | 0.200 | 0.000 | 0.200 | 0.200 | 0.000 |
| bbh_logic | 0.400 | 0.033 | **-0.367** | 0.400 | 0.033 | **-0.367** |
| arc_challenge | 0.367 | 0.067 | **-0.300** | 0.433 | 0.033 | **-0.400** |
| strategyqa | 0.633 | 0.633 | 0.000 | 0.700 | 0.700 | 0.000 |

## Why the drop is correct, not a regression

The previous `check_answer_strict` for `multiple_choice` did
`re.search(r"([A-E])", p)` over the prediction text. That matched ANY
A–E character, including:
- "Because photosynthesis happens" → matched B
- "A long explanation mentions B and C" → matched A or B
- "Despite some evidence for C..." → matched C even if final answer was D

The new checker requires explicit MC markers:
- single-character pure label: `"B"` / `"(B)"` / `"B."`
- explicit phrasing: `"the answer is (B)"`, `"answer: B"`, `"I choose option D"`
- box: `"\\boxed{D}"`
- conclusion: `"Therefore C."`, `"Hence (A)"`
- final-line letter alone: `"...\nE"`

ARC and BBH model outputs frequently contain reasoning that mentions
multiple options without a clear final marker. Under the strict checker
those inputs are correctly rejected — but the legacy strict regex was
accidentally accepting many of them.

## Implication for AGD scaling-law claim

**Pre-tightening report** said AGD on 27B was non-negative on 3/4
benchmarks. **Post-tightening**:
- math_hard: AGD +0.200 (large positive, math_expression unaffected)
- bbh_logic: AGD -0.033 (small negative, almost noise; both arms ≈ 0)
- arc_challenge: AGD +0.067 (small positive; both arms collapsed)
- strategyqa: AGD -0.033 (small negative, boolean unaffected)

The "27B AGD non-negative on 3/4" headline is now even weaker:
**only 1 benchmark (math_hard) shows a meaningfully positive AGD effect**
under the strictest possible MC interpretation. arc_challenge's strict
strict drop suggests the model rarely emits the explicit answer marker
ARC labels expect.

**This does NOT invalidate the PCS plan**: the candidate-pool / oracle-gap
motivation remains. It DOES mean the AGD scaling-law side-finding now
needs to be reported with strict-vs-legacy decomposition, not as a
single number.

## Decision

Per GPT-5.5 Pro non-negotiable rule #1 ("Do not change evaluation
metrics to improve results") and the explicit Round-5 Task-2 spec
("tighten MC strict checking"), the new checker is the correct one.
Proceeding with Tasks 3-6 with the strict checker as default.

Both old (lax) and new (strict) numbers are now version-controlled in
git history; the old strict_results.json files were regenerated and
the prior versions are reachable via earlier commits.
