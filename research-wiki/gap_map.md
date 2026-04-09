# Field Gap Map

## G1: Topology → Strategy Routing [PARTIALLY ADDRESSED]
Nobody uses reasoning trace topology to predict optimal inference strategy.
- Shape of Reasoning (TDA) analyzes but doesn't route
- DiffAdapt routes via hidden states, not topology
- CLOX v2 attempted but data didn't support claims
- Status: Gap exists but may be unfillable with current models

## G2: Per-Instance Test-Time Scaling Law [SATURATED]
No Chinchilla-equivalent for inference-time compute.
- Snell et al. showed difficulty-dependent allocation
- Entropy trajectory papers (12+) cover most angles
- Status: Crowded space, hard to differentiate

## G3: Self-Correction Without Training [OPEN]
Huang et al. barrier unbroken without RL training.
- SCoRe uses RL (requires training)
- No training-free approach shown to work
- Structural signals (CLOX) are circular (same model)
- Status: Open but extremely hard

## G4: Zero-Cost Process Reward [PARTIALLY ADDRESSED]
Trained PRMs are expensive. No free alternative.
- Self-Certainty (KL-div) achieves ~80% of PRM
- DeepConf (confidence windows) works for R1-style models
- Entropy-based approaches are cheap but simple
- Status: Addressed by Self-Certainty, but richer signals unexplored

## G5: Selection vs Voting Theory [OPEN - BEST OPPORTUNITY]
When should you pick the best trace vs majority vote?
- Di et al. (BoM, ICLR 2026): minimax framework, but worst-case not per-instance
- Wu et al.: empirical but no crossover formula
- Chen et al.: non-monotonic voting but no selection comparison
- Status: Open. Per-instance crossover K* formula does not exist.

## G6: Cross-Strategy Diversity [OPEN]
Using structurally different strategies as one-model ensemble.
- Div-Se (2023): different approaches via prompts only
- FOBAR (2024): K=2 only (forward+backward), math-specific
- CoTnPoT (2024): K=2 only (CoT+code), math-specific
- Status: General K-strategy framework with independence analysis is novel.

## G7: MCTS Degradation at High Budgets [PARTIALLY ADDRESSED]
ReSCALE fixed with better sampling, but no structural signal.
- AB-MCTS (NeurIPS 2025): adaptive wider/deeper
- Status: Topology-guided MCTS unexplored but implementation-heavy

## G8: Late-Stage Fragility Theory [PARTIALLY ADDRESSED]
ASCoT found empirically but no theoretical explanation.
- Manifold-width theory could explain it
- Status: Theory is novel but empirical observation is claimed
