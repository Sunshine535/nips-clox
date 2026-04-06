# Plan: CLOX (AutoResearchClaw 23-Stage Pipeline)

## Phase A: Research Scoping (Stages 1–2) — Complete

- [x] **Stage 1: Topic Refinement**
  Refined topic from broad "partial masking for LLM reasoning" to focused investigation of inference-time cloze-style reasoning on decoder-only models.

- [x] **Stage 2: SMART Goal**
  Produced `/docs/goal.md`: 8-week study targeting +2.0 accuracy or 25% token savings on GSM8K/MATH500/BBH using Cloze-of-Thought (CoT*) with 7B–8B open models.

## Phase B: Literature Discovery (Stages 3–6) — Complete

- [x] **Stage 3: Broad Literature Search**
  Searched across chain-of-thought, masked language modeling, fill-in-the-middle, self-consistency, MoE routing, and test-time compute literatures.

- [x] **Stage 4: Paper Screening**
  Screened papers for direct relevance to inference-time partial masking of reasoning traces. Retained 11 core references.

- [x] **Stage 5: Deep Reading**
  Produced detailed reading notes on Wei et al. (CoT), Wang et al. (self-consistency), Yao et al. (Tree of Thoughts), Zhou et al. (least-to-most), Du et al. (GLM blank infilling), and 6 additional papers.

- [x] **Stage 6: Gap Analysis**
  Identified 7 gaps in `/docs/synthesis.md`: no direct cloze-reasoning benchmark, no unified framework connecting token/rationale/MoE masking, weak understanding of accuracy vs efficiency tradeoffs.

## Phase C: Knowledge Synthesis (Stages 7–8) — Complete

- [x] **Stage 7: Literature Synthesis**
  Produced `/docs/synthesis.md` with 3 clusters (masked reasoning generation, MoE selective computation, error-aware training) and 7 prioritized opportunities.

- [x] **Stage 8: Problem Decomposition**
  Produced `/docs/problem_tree.md` with 8 sub-questions, priority ranking, and 8 risks with mitigations.

## Phase D: Experiment Design (Stages 9–11) — Complete

- [x] **Stage 9: Hypothesis Formulation**
  Produced `/docs/hypotheses.md` with 3 final hypotheses synthesizing innovator/contrarian/pragmatist perspectives. Recommended experimental order: H2 first (best falsifiability), H1 second, H3 third.

- [x] **Stage 10: Experiment Plan**
  Produced `/docs/experiment_plan.yaml` specifying 6 conditions, 4 datasets, 5 seeds, 300s runtime budget, token budgets per condition, and statistical reporting requirements.

- [x] **Stage 11: Plan Review**
  Validated experiment plan against hypotheses. Confirmed compute feasibility on single GPU with 4-bit quantized model.

## Phase E: Experiment Execution (Stages 12–13) — Complete

- [x] **Stage 12: Implementation**
  Implemented all 6 conditions in `/code/methods/`:
  - StandardLeftToRightChainOfThought
  - BudgetMatchedSelfConsistencyConsensus
  - FullRationaleRegenerationRevision
  - AnswerAnchoredBackwardClozeReconstruction
  - UncertaintyTargetedSelectiveMaskedRepair
  - RandomSpanMaskedRepair
  Plus 3 ablations: ForwardOrderCloze, WholeRationaleMaskRepair.

- [x] **Stage 13: Experiment Execution**
  Ran all conditions on synthetic seven-segment benchmark (64 examples × 5 seeds × 6 conditions). Results stored in `/results/`.

## Phase F: Analysis & Decision (Stages 14–15) — Complete

- [x] **Stage 14: Result Analysis**
  Produced `/analysis/result_analysis.md` with metrics summary, consensus findings, contested points, statistical checks, methodology audit, and quality rating (4/10).

- [x] **Stage 15: Decision Gate**
  Quality gate: **2.3/4.0 (degraded mode)**. Decision: **REFINE** (not PROCEED, not full PIVOT).
  - Flagship backward-cloze hypothesis not supported.
  - Self-consistency outperformed all masking variants.
  - Two key ablations invalid (identical outputs).
  - Result classified as negative-but-informative.

## Phase G: Paper Writing (Stages 16–19) — Complete

- [x] **Stage 16: Paper Outline**
  Produced `/docs/outline.md` with positioning decision, thesis paragraph, and detailed section-by-section outline.

- [x] **Stage 17: Figure Generation**
  Generated method diagram, results comparison table, and confidence-split visualization in `/paper/figures/`.

- [x] **Stage 18: Section Drafting**
  Drafted all sections in `/paper/sections/`: introduction, related work, method, experiments, discussion.

- [x] **Stage 19: Paper Assembly**
  Assembled full paper draft with references from `/paper/references.bib`.

## Phase H: Finalization (Stages 20–23) — Complete

- [x] **Stage 20: Internal Review**
  Reviewed draft for consistency, overclaim avoidance, and statistical rigor. Ensured negative-result framing is clear.

- [x] **Stage 21: Quality Gate Assessment**
  Final quality gate: 2.3/4.0. Paper positioned as negative-but-informative empirical study.

- [x] **Stage 22: Reproducibility Package**
  Packaged code, configs, and seeds for full reproducibility. Per-example predictions and per-token logprobs saved.

- [x] **Stage 23: Export**
  Exported deliverables: paper draft, references, experiment code, results, and analysis.

## Kill Criteria

- If no masking-specific benefit survives compute-matched comparison after replication on real benchmarks, pivot framing to:
  - inference-time revision/repair as a general paradigm,
  - diagnostic study of when partial masking harms vs helps,
  - negative result with strong reproducibility contribution.
