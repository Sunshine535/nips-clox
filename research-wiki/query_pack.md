# Query Pack (auto-generated 2026-04-09)

## Project: CLOX — Inference-time compute optimization for LLM reasoning
Hardware: 2xH100, vLLM, Qwen3-32B-AWQ. Target: NeurIPS 2025.

## Top 5 Gaps
G5: Selection vs voting crossover theory — no per-instance K* formula exists [OPEN]
G6: Cross-strategy diversity for verification — general K-strategy framework unexplored [OPEN]
G3: Self-correction without training — Huang barrier unbroken [OPEN but extremely hard]
G1: Topology→strategy routing — attempted by CLOX v2, failed empirically [HARD]
G2: Per-instance scaling law — saturated with 12+ entropy papers [CROWDED]

## Failed Ideas (BANLIST)
idea:001 CLOX v2 topology→strategy: ALL benchmarks short EPL, repair collapses at scale, SC dominates
idea:002 Entropy trajectory scaling: SCOOPED by Zhao2026 + 11 others
idea:003 Structural verification: 4/10 review, data contradicts, "zero-cost" dishonest, faithfulness unprovable

## Paper Clusters
SCALING: Snell2024(ICLR2025), Wu2024(ICLR2025), Brown2024, Liu2025(ICLR2025), Roberts2026 — establish compute-optimal TTS
TOPOLOGY: Tan2025(TDA), Chen2026(molecular), Xiong2025(graph) — analyze trace structure, never close the loop
SELECTION: Di2026(BoM,ICLR2026), Kang2025(SelfCertainty), DeepConf2025 — scoring for BoN selection
DIVERSITY: DivSe2023, FOBAR2024, CoTnPoT2024, Dipper2024 — limited cross-strategy/prompt diversity

## Active Ideas
idea:005 Selection-voting crossover (PARTIAL novelty, vs BoM)
idea:006 Cross-strategy verification (PARTIAL novelty, vs DivSe/FOBAR)

## Key Constraint
SC-5 beats ALL structural methods at n=200. Any proposal claiming to beat SC must explain why our data is different or test in a different regime.
