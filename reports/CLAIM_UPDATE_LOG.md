# Claim Update Log

Each claim is either **retained**, **weakened**, or **retracted** relative to
the repo state before GPT-5.5 Pro review.

## Retracted — not made in any current or future paper

| Claim | Where it lived | Reason |
|---|---|---|
| "Topology r̄ and ℓ determine optimal reasoning strategy" | `paper/main.tex` abstract, older README | GPT-5.5 Pro §5 P11: topology correlations ≈ 0 with disagreement/diversity. Pilot-to-scale collapse. |
| "CLOX saves 40–60% compute while matching SC" | early README | Adaptive eval never completed; §R9 in GPT-5.5 Pro result table marked Missing Log. |
| "Masked repair beats SC in high-recoverability regimes" | paper theorem | §P01/P02: targeted=random ablation collapse; repair prompts allow full rewrite. |
| "No fixed strategy dominates" (as a theorem) | paper | Empirically SC dominates several visible settings; not refuted by current data. |
| "First adaptive strategy selector" | implied in draft | Close prior art (Self-Certainty, RTR, PRISM). |

## Weakened — keep but narrow scope

| Claim | From | To |
|---|---|---|
| "CLOX-Adaptive near-oracle" | headline | "CLOXAdaptive is a LEGACY_BASELINE used for historical comparison only" (see `docs/legacy_status.md`). |
| "AGD universally helps diverse prompting" | internal | "AGD is universally non-negative on 27B across 4 benchmarks (avg +4.2%, min +0%); fails on 9B (avg -6.7%, min -23.3%). Finding is a capability-dependent scaling law, not a universal claim." |
| "Oracle gap proves selection opportunity" | implied | "Oracle-SC gap up to 33.3% on 27B indicates the candidate pool contains correct answers not selected by majority vote. Labels are used; cannot be deployed as a method." |
| "Backward cloze and BAV help" | BAV section | "BAV shows Pareto-efficiency signal (0.80 @ 7522 tokens vs SC 0.88 @ 17360) but agreement gate is non-discriminative (agreed=disagreed=80%). Kept as ablation." |
| "Cross-strategy vote beats SC" | IDEA_REPORT | "Cross-strategy candidate pool has oracle room (oracle_any 0.90 vs SC 0.88) but naive voting underperforms SC (cross_K5 0.82). Requires calibrated selection." |

## Retained — evidence is solid

| Claim | Evidence |
|---|---|
| "Oracle-SC gap is real and non-trivial on several model/task cells" | `results/meta/meta_summary.json`; 11 cells ≥ 15% gap, max 33.3% |
| "Diverse prompt candidates produce informative disagreement on strong models" | AGD 27B 4/4 non-negative, 0.73x tokens |
| "Weaker models cannot exploit prompt diversity" | AGD 9B -23% on math_hard |
| "BAV agreement gate is non-discriminative" | `results/bav/bav_analysis.json` agreed=disagreed=0.80 |
| "Early ablations identical across 9 metrics" | `EXPERIMENTS.md` self-reported |

## New claims (only made after PCS empirical run)

These claims will only be written into the paper if the corresponding
experiment result meets the pre-registered threshold:

- "CLOX-PCS converts the oracle-SC gap into deployable accuracy/token Pareto improvement on [benchmark list] at matched budget."
  Threshold: C > B by ≥ 2 pts OR iso-accuracy with ≥ 20% token saving, paired CI excludes 0, 3 seeds.
- "Calibrated selector learns transferable signal from cross-strategy features; topology proxies contribute marginally."
  Threshold: ablation without topology features does not degrade AUC by > 0.02.
- "Value-of-compute gate reduces token cost without significant accuracy loss."
  Threshold: C with gate saves ≥ 10% tokens vs C without gate, accuracy within 1 point.

## How to add / retract a claim

1. Implement the claim in code first.
2. Run the corresponding experiment with pre-registered success criterion.
3. If it passes, promote to paper + log here with the supporting result file path.
4. If it fails, log the failure here and do NOT write the claim.

A claim present in this log without a result file reference is **not paper-ready**.
