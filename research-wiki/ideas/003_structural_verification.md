---
type: idea
node_id: idea:003
title: "Zero-Cost Structural Verification (Breaking Self-Correction Barrier)"
stage: killed
outcome: reviewed_reject
target_gaps: [G3, G4]
created_at: 2026-04-09
---

# Structural consistency as zero-cost PRM and self-correction enabler

## Hypothesis
Composite score S(trace) = FB + SI + ES serves as zero-cost PRM and breaks self-correction barrier.

## External Review: 4/10
### Fatal weaknesses:
1. S >= P(correct) requires faithfulness assumption, which is empirically false
2. Own data: SC-5 (87%) crushes targeted_repair (70.5%) — structural signals lose to brute voting
3. "Zero-cost" misleading: FB and SI cost more than 5-10 SC samples
4. Self-correction claim: S is computed by same model = not truly external

## Lesson
- Don't claim to replace PRMs unless you can actually beat them empirically
- Selection is easier than repair but still needs the signal to correlate with correctness
- "Zero-cost" must be honest — only entropy from forward pass is truly zero-cost
- The faithfulness assumption is a trap — avoid any theorem that requires it
