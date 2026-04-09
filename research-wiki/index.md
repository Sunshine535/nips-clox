# Research Wiki Index

## Papers

### Test-Time Compute Scaling
- [paper:snell2024_ttc](papers/snell2024_ttc.md) — Compute-optimal test-time scaling, 4x over BoN
- [paper:wu2024_inference_scaling](papers/wu2024_inference_scaling.md) — Empirical inference scaling laws across strategies
- [paper:brown2024_monkeys](papers/brown2024_monkeys.md) — Repeated sampling scales log-linearly, selection bottleneck
- [paper:liu2025_1b_surpass](papers/liu2025_1b_surpass.md) — 0.5B beats GPT-4o with compute-optimal TTS
- [paper:roberts2026_t2](papers/roberts2026_t2.md) — T-squared: joint train-test scaling laws

### Reasoning Topology
- [paper:tan2025_shape](papers/tan2025_shape.md) — TDA of reasoning traces, topological features predict quality
- [paper:chen2026_molecular](papers/chen2026_molecular.md) — Molecular structure of thought, 3 bond types
- [paper:xiong2025_graph](papers/xiong2025_graph.md) — Graph-based analysis of reasoning structure

### Self-Correction & Verification
- [paper:huang2024_selfcorrect](papers/huang2024_selfcorrect.md) — LLMs cannot self-correct without external signal
- [paper:kumar2024_score](papers/kumar2024_score.md) — SCoRe: RL-based self-correction
- [paper:jiang2024_fobar](papers/jiang2024_fobar.md) — Forward-backward verification for math
- [paper:kang2025_selfcertainty](papers/kang2025_selfcertainty.md) — Zero-cost BoN selection via self-certainty

### Entropy & Uncertainty
- [paper:zhao2026_entropy_trajectory](papers/zhao2026_entropy_trajectory.md) — Entropy trajectory shape predicts reliability
- [paper:li2025_entropy_gated](papers/li2025_entropy_gated.md) — Entropy-gated branching, +22.6% accuracy
- [paper:sun2026_reinforcement_inference](papers/sun2026_reinforcement_inference.md) — Entropy-aware adaptive inference

### Strategy Selection & Routing
- [paper:damani2025_adaptive](papers/damani2025_adaptive.md) — Input-adaptive compute allocation (ICLR 2025)
- [paper:di2026_bom](papers/di2026_bom.md) — Best-of-Majority: minimax-optimal (ICLR 2026)
- [paper:inoue2025_abmcts](papers/inoue2025_abmcts.md) — AB-MCTS wider vs deeper (NeurIPS 2025 Spotlight)

### Efficient Reasoning
- [paper:aggarwal2025_l1](papers/aggarwal2025_l1.md) — Length-controlled reasoning via RL
- [paper:li2025_selfbudgeter](papers/li2025_selfbudgeter.md) — Adaptive token budget per difficulty
- [paper:ascot2025](papers/ascot2025.md) — Late-stage fragility, positional error impact

### Process Rewards
- [paper:khalifa2025_thinkprm](papers/khalifa2025_thinkprm.md) — Generative PRM with 1% labels
- [paper:setlur2024_pavs](papers/setlur2024_pavs.md) — Process Advantage Verifiers

## Ideas

- [idea:001](ideas/001_clox_v2_topology_strategy.md) — CLOX v2: Topology predicts strategy [FAILED]
- [idea:002](ideas/002_entropy_trajectory_scaling.md) — Entropy trajectory as scaling law [SCOOPED]
- [idea:003](ideas/003_structural_verification.md) — Zero-cost structural verification [REVIEWED: 4/10]
- [idea:004](ideas/004_late_stage_fragility.md) — Topological theory of late-stage fragility [PARTIAL]
- [idea:005](ideas/005_selection_vs_voting.md) — Selection-voting crossover theory [ACTIVE]
- [idea:006](ideas/006_cross_strategy_verification.md) — Cross-strategy ensemble verification [ACTIVE]

## Experiments

- [exp:001](experiments/001_clox_v2_topology.md) — CLOX v2 topology characterization
- [exp:002](experiments/002_clox_v2_strategies.md) — CLOX v2 strategy comparison (partial)
- [exp:003](experiments/003_synthetic_dag.md) — Synthetic DAG validation (83% accuracy)

## Claims

- [claim:C1](claims/C1_topology_predicts_strategy.md) — Topology (r-bar, ell) predicts optimal strategy [INVALIDATED]
- [claim:C2](claims/C2_short_epl_universal.md) — All real benchmarks have short EPL [SUPPORTED]
- [claim:C3](claims/C3_sc_dominates_short_epl.md) — SC dominates in short-EPL regime [SUPPORTED]

## Gaps

See [gap_map.md](gap_map.md)
