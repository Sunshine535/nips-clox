---
type: idea
node_id: idea:001
title: "CLOX v2: Topology Predicts Optimal Inference Strategy"
stage: failed
outcome: negative
target_gaps: [G1]
created_at: 2026-04-05
---

# CLOX v2: Topology (r-bar, ell) predicts optimal inference strategy

## Hypothesis
Two measurable properties — local recoverability (r-bar) and error propagation length (ell) — partition reasoning tasks into regimes where different strategies are optimal.

## What We Built
- Topology estimator (r-bar, ell) from pilot traces
- 9 inference strategies (CoT, SC, repair, backward cloze, etc.)
- Synthetic DAG validation framework
- Full experiment pipeline on 2xH100

## What Happened
- Synthetic DAG: 83% prediction accuracy (POSITIVE)
- Real benchmarks: ALL have short EPL (ell ≈ 1.2-1.6) — no regime diversity
- SC-5 dominates everywhere (87% GSM8K vs repair 70.5%)
- targeted_repair collapsed from 98% (n=50) to 70.5% (n=200)
- StrategyQA at chance (50%) regardless of strategy
- Review: R1 2/10, R2 5.5/10

## Why It Failed
1. Theory assumes two discrete regimes but real data shows a continuum
2. All benchmarks cluster in the same topology region
3. Repair requires faithful local recovery which doesn't scale
4. Pilot-to-scale generalization fails (n=50 overfits to easy examples)

## Lessons
- Topology IS measurable and varies across tasks (validated)
- But topology doesn't predict strategy winners at scale (invalidated)
- SC is remarkably robust — hard to beat with structural methods
- Don't trust pilot results at n=50; need n>=200 minimum
