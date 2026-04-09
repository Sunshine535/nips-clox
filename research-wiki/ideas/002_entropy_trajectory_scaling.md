---
type: idea
node_id: idea:002
title: "Entropy Trajectory as Per-Instance Scaling Law"
stage: killed
outcome: scooped
target_gaps: [G2]
created_at: 2026-04-09
---

# Entropy trajectory shape as sufficient statistic for test-time compute

## Hypothesis
Per-token entropy trajectory shape predicts optimal token budget and strategy.

## Why Killed
SCOOPED by 12+ papers (2024-2026):
- Zhao 2026 (arXiv:2603.18940): entropy trajectory monotonicity predicts correctness
- EAS 2025 (arXiv:2508.20384): integral of entropy as metric
- Adaptive-Consistency 2023: entropy-based adaptive K for SC
- DiffAdapt 2025: U-shaped entropy for difficulty routing
- Reinforcement Inference 2026: entropy-triggered re-attempts
- Entropix: entropy+varentropy for strategy switching
- EGSS 2026: entropy-guided stepwise scaling
- PREGU 2026: entropy-triggered partial repair

## Lesson
The "entropy for LLM reasoning" space is completely saturated as of 2026. Any new paper must go beyond entropy as the primary signal.
