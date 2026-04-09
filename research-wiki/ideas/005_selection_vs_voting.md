---
type: idea
node_id: idea:005
title: "Selection-Voting Crossover: When to Pick vs When to Vote"
stage: active
outcome: pending
target_gaps: [G5]
created_at: 2026-04-09
novelty_verdict: partial
---

# When Selection Beats Voting: Per-Instance Crossover Theory

## Hypothesis
A crossover K* = f(pass_rate, scorer_AUC) exists where trace selection outperforms majority voting. Below K*, even crude scorers beat voting.

## Novelty Status: PARTIAL
- Di et al. (BoM, ICLR 2026): minimax framework, but worst-case not per-instance
- Wu et al.: empirical but no crossover formula
- Chen et al.: non-monotonic voting but no selection comparison
- NOVEL: the specific K* formula and practitioner-facing decision rule

## Differentiation vs BoM (main competitor)
- BoM: worst-case minimax for pass@k setting
- Ours: instance-adaptive decision rule parameterized by (pass_rate, scorer_AUC)
- BoM doesn't give practitioners a concrete "when" answer; we do

## Risks
- Theory is straightforward (voting fails at low pass rate is known)
- Needs strong experimental validation showing the crossover is sharp
- May not reach "best paper" level — solid 6-7/10 contribution
