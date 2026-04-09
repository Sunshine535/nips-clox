---
type: idea
node_id: idea:006
title: "Cross-Strategy Verification: One-Model Ensemble via Strategy Diversity"
stage: active
outcome: pending
target_gaps: [G6]
created_at: 2026-04-09
novelty_verdict: partial
---

# Cross-Strategy Verification: Structural Diversity for Ensemble Reasoning

## Hypothesis
Running K different strategies (CoT, backward cloze, repair, regen) produces ε-independent errors, enabling cross-strategy majority vote to beat single-strategy SC at matched token budget.

## Novelty Status: PARTIAL
- Div-Se (2023): diverse approaches via prompts only, no independence analysis
- FOBAR (2024): K=2 (forward+backward), math-specific
- CoTnPoT (2024): K=2 (CoT+code), math-specific
- Dipper (2024): one-model ensemble via prompt diversity only
- NOVEL: general K-strategy framework with independence analysis + topology-based diversity

## Key Advantage
Directly leverages ALL of CLOX v2's infrastructure (9 strategies already implemented!). The question is whether strategies produce genuinely independent errors.

## Risks
- Strategies may share failure modes (same model = correlated errors)
- Token budget matching is tricky (strategies have very different costs)
- "Consensus is Not Verification" (2026) shows within-model consensus can be illusory
- Need to prove ε-independence empirically
