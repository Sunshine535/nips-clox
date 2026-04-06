# Paper Planning Outline

## Working method name
**CLOX**  
_short for **clo**ze-style inference with masked repair_.  
Why this name works: 4 chars, easy to remember, directly signals the central idea of partial masking / fill-in-the-blank reasoning.

---

## Candidate titles

| Title | Memorability | Specificity | Novelty signal | Notes |
|---|---:|---:|---:|---|
| **CLOX: Partial Masking for Inference-Time Reasoning in Language Models** | 4/5 | 5/5 | 4/5 | Best balanced title; broad enough to cover cloze, repair, and revision |
| **CLOX: Can Fill-in-the-Blank Reasoning Improve LLM Inference?** | 5/5 | 4/5 | 4/5 | Strong hook; slightly more question-driven |
| **CLOX: Evaluating Cloze-Style Thought Reconstruction for LLM Reasoning** | 4/5 | 5/5 | 3/5 | Most precise for the current pilot, but narrower and less punchy |

**Recommended title:**  
**CLOX: Partial Masking for Inference-Time Reasoning in Language Models**

---

# Positioning decision

Because the current evidence does **not** support the strongest version of the headline claim, the paper should be framed as:

> a **careful empirical study of partial masking at inference time**, rather than a triumphant method paper claiming a clear win.

That framing is much more defensible for NeurIPS/ICLR/ICML style writing. The central message should be:

- partial masking and cloze-style restructuring are **feasible** inference-time controls,
- they do **not automatically improve** reasoning,
- in this pilot, **self-consistency outperformed masking-specific variants**,
- therefore the field should treat masking as a **hypothesis to test**, not an established improvement mechanism.

This still fits the required topic tightly.

---

# Paper one-paragraph thesis

This paper asks whether large language models can benefit from **partial masking at inference time**—for example, by reconstructing a rationale in cloze form, selectively repairing uncertain spans, or conditioning reasoning on answer anchors. We introduce the **CLOX** framework, a unifying view of inference-time masking and rationale reconstruction, and compare backward cloze reconstruction, selective masked repair, random masking, full rationale regeneration, and budget-matched self-consistency. On a pilot seven-segment reasoning benchmark, masking-based methods remain competitive with standard chain-of-thought but do **not** outperform it reliably; the best mean result comes from self-consistency rather than the flagship cloze method. The paper therefore contributes a more precise conclusion: **inference-time restructuring appears viable, but masking-specific gains remain unproven**.

---

# Detailed outline

## 0. Front matter

### Title
Use one of the candidate titles above.

### Abstract
**Target:** 190–210 words  
**Goal:** State the research problem clearly, introduce CLOX by sentence 3, report the main quantitative outcome, and make the negative-but-informative takeaway crisp.

**What the abstract must do**
- Sentence 1–2: state the gap:
  - LLM reasoning is usually left-to-right.
  - It is unclear whether **partial masking / cloze-style inference** can improve reasoning by reducing error propagation.
- Sentence 3–4: introduce **CLOX** as a unifying inference-time framework for:
  - answer-anchored backward cloze reconstruction,
  - selective masked repair,
  - random span repair,
  - full rationale regeneration,
  - budget-matched self-consistency comparison.
- Sentence 5–6: give concrete numbers:
  - Standard CoT: **75.52%**
  - Backward cloze: **75.52%**
  - Best method, budget-matched self-consistency: **77.60%**
  - Selective and random masked repair: **76.04%**
- Final sentence: emphasize supported conclusion:
  - masking-based restructuring is feasible,
  - but the present pilot does not validate masking as the causal source of gains.

**Evidence links**
- Main results table: baseline 75.52%, backward cloze 75.52%, SC 77.60%, repair 76.04–76.56%
- Consensus Finding 1: flagship hypothesis not supported
- Consensus Finding 3: self-consistency best
- Limitations: pilot-scale, 64 examples, 3 seeds

---

## 1. Introduction
**Target:** 850–950 words  
**Goal:** Motivate the problem, define the exact question, situate it in reasoning-time control for LLMs, and present the paper as a disciplined empirical study rather than an overclaim.

### Paragraph 1 — Motivation
**Purpose:** Explain why left-to-right reasoning may be suboptimal for some tasks.  
Key themes:
- Autoregressive reasoning commits early and may propagate mistakes.
- Many tasks are naturally solvable with **fill-in-the-blank**, local repair, or answer-conditioned verification.
- This motivates asking whether inference can be reorganized into a partially masked process.

**Core sentence to aim for**
> Standard chain-of-thought commits to a single left-to-right trajectory, but many reasoning problems admit alternative decompositions in which uncertain intermediate content can be reconstructed, repaired, or inferred from context.

### Paragraph 2 — Gap in prior work
**Purpose:** Explain what is missing in the literature.  
You should contrast:
- chain-of-thought,
- self-consistency,
- verifier/revision methods,
- masked language modeling intuition,
- but note that **partial masking during inference for causal LLM reasoning** is under-tested.

**Citations to include**
- Chain-of-thought: Wei et al.
- Self-consistency: Wang et al.
- Least-to-most / decomposition: Zhou et al.
- ReAct / tool-based reasoning: Yao et al.
- Reflexion / revision: Shinn et al.
- Verifier-style or debate-style refinement papers
- Masked modeling / span corruption references: Devlin et al., Raffel et al., Lewis et al.
- Recent reasoning control papers from 2024–2025 if available

### Paragraph 3 — Our framing and approach
**Purpose:** Introduce **CLOX** and define the tested family of methods.  
Describe CLOX as a framework rather than a single algorithmic trick:
- answer-anchored backward cloze reconstruction,
- uncertainty-targeted masked repair,
- random span repair,
- full rationale regeneration,
- budget-matched self-consistency as a strong non-masking comparator.

The paragraph should make clear that the paper tests whether **partial masking helps at any inference stage**, especially within rationale reconstruction and repair.

### Paragraph 4 — Main empirical message
**Purpose:** Tell the reader what happened without overselling.  
State:
- masking-based methods are competitive,
- backward cloze did not beat baseline,
- self-consistency achieved the best mean score,
- key masking ablations require caution.

### Paragraph 5 — Contributions paragraph
Keep as prose, not bullets in final draft, though planning can list them here.

**Contribution content**
1. A unifying formulation of **inference-time partial masking** for LLM reasoning.
2. A controlled pilot comparison across cloze reconstruction, selective repair, random repair, rationale regeneration, and self-consistency.
3. An empirical finding that **viability does not imply advantage**: cloze-style masking does not outperform baseline in the present setting.
4. A methodological takeaway that masking claims require stronger ablations, paired tests, and budget accounting.

**Evidence links**
- Research question from prompt: partial masking during inference / CoT / MoE-style activation
- Main results summary
- Methodology audit: need to avoid causal overclaim
- Decision: REFINE, not PROCEED

---

## 2. Related Work
**Target:** 650–800 words  
**Goal:** Organize prior work into 3 subsections and end each with how CLOX differs.

### 2.1 Inference-time reasoning for LLMs
**Scope**
- Chain-of-thought
- Self-consistency
- Tree-of-thought / search
- Least-to-most