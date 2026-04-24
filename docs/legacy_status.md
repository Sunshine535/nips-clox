# Legacy Status

## CLOXAdaptive / Topology-Threshold Routing — LEGACY_BASELINE

**Status**: FROZEN. Not the current main method path.

The original CLOX v2 hypothesis was that topology estimates (r̄, ℓ) could predict optimal inference strategy. This has been **contradicted** by:

1. Topology correlation with disagreement/diversity near zero (pilot analysis)
2. Pilot-to-scale collapse: targeted_repair 98% n=50 → 70.5% n=200
3. `CLOXAdaptive` mostly receives "adaptive" recommendation, which maps to `targeted_repair` — selector degenerates
4. All real benchmarks have short ℓ; SC-dominant regime absent

**Use as**: legacy baseline only. Do not use for new method claims.

## Targeted/Random Repair — ABLATION ONLY

Both `UncertaintyTargetedRepair` and `RandomRepair` produce near-identical results (early ablation and pilot evidence). The repair prompt allows full solution rewrite, not local masked repair. Keep as candidate generators / ablations only.

## BAV (Backward-Anchored Verification) — BASELINE

BAV sits on the Pareto front (80% @ 7.5k tokens) but does not beat SC (88%). The agreement gate is non-discriminative (agreed=disagreed=80%). Keep as baseline for PCS comparison.

## New Main Path: CLOX-PCS

Per GPT-5.5 Pro diagnosis, the new main method is **CLOX-PCS (Calibrated Portfolio Compute Selection)**:
- Existing strategies become candidate generators
- Calibrated selector learns which candidate is correct
- Value-of-compute gate controls budget
