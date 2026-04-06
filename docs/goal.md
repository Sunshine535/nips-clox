# SMART Research Goal

## **Topic**
**Inference-time partial masking for LLM reasoning** — investigating whether decoder-only large language models can improve reasoning accuracy, robustness, or efficiency by being forced to solve parts of a problem in a **cloze / fill-in-the-blank** manner during inference, rather than generating a fully free-form chain of thought left-to-right.

---

## **Novel Angle**
### **Core idea**
Study a new inference-time method, **Cloze-of-Thought (CoT\*)**, where an LLM is prompted to:
1. produce a **sparse reasoning scaffold**,
2. leave selected spans as **blanks**,
3. then iteratively fill those blanks using the model’s own conditional reasoning.

The research question is not “does masking help pretraining?” or “does chain-of-thought help?”, both of which are already well studied. The unexplored angle is:

> **Can partial masking of intermediate reasoning spans at inference time reduce error propagation and improve final-task performance under a fixed token budget?**

### **Why this is NOT already well-covered**
This direction is distinct from several nearby literatures:

- **Masked language modeling / span corruption**: these are primarily **pretraining objectives** for bidirectional or encoder-decoder models, not inference-time reasoning strategies for decoder-only LLMs.
- **Fill-in-the-middle (FIM)**: mostly studied for **code completion/editing**, not for mathematical or symbolic reasoning traces where the model must generate and resolve missing intermediate steps.
- **Standard Chain-of-Thought (CoT)**: generates a full sequential rationale; it does **not** intentionally hide uncertain substeps and solve them as constrained blanks.
- **Self-consistency / reranking / tree search**: these sample many complete reasoning paths, but do not test whether **structured omission** of reasoning spans itself improves reasoning.
- **MoE routing work**: focuses on expert selection efficiency or load balancing, not on **reasoning-aware partial masking** of intermediate content. MoE can be discussed as a future extension, but it should not be the core claim of a first paper.

### **Why this is timely NOW (2024-2026)**
This is timely because several 2024 trends create a clear opening:

1. **Inference-time reasoning is now a major frontier.**  
   Recent work shows that *how* a model reasons at test time matters nearly as much as parameter count.
2. **Reasoning verbosity is increasingly a liability.**  
   Long CoT is costly, brittle, and may reveal unfaithful or unstable rationales. A sparse blank-filling format could reduce unnecessary token generation.
3. **Open 7B-8B models are now strong enough** to test structured inference interventions without expensive retraining.
4. **Recent papers suggest reasoning structure matters**, but they still leave a gap around **partial masking of reasoning traces**.

### **Trend validation: recent papers (2024)**
These papers establish that structured test-time reasoning is an active and relevant direction:

- **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking (2024)**  
  Relevance: shows that modifying how intermediate reasoning is represented can improve performance.
- **Self-Discover: Large Language Models Self-Compose Reasoning Structures (2024)**  
  Relevance: shows that reasoning *structure selection* at inference time matters.
- **Chain of Draft: Thinking Faster by Writing Less (2024)**  
  Relevance: shows that shorter, more structured reasoning can retain or improve performance while reducing token cost.

### **How this differs from standard approaches**
Standard CoT asks the model to generate all reasoning steps left-to-right.  
This project instead asks:

- Which reasoning spans should remain **explicitly blank**?
- Should blanks be chosen **randomly**, by **entropy/uncertainty**, or by **step type**?
- Does solving reasoning as a set of local cloze problems reduce cascading errors?

That is a sharper and less-explored question than generic “prompt engineering for reasoning.”

---

## **Scope**
A focused single-paper scope:

**Primary focus:**  
Inference-time partial masking of reasoning traces for **math and symbolic reasoning** in open instruction-tuned LLMs.

**In scope**
- Decoder-only open models in the 3B-8B range
- Prompt-only or very light adaptation
- Cloze-style reasoning scaffolds with iterative blank filling
- Comparison against:
  - direct answer
  - standard CoT
  - concise reasoning baselines (e.g., Chain-of-Draft-style prompting)
- Ablations on:
  - masking policy
  - masking ratio
  - one-shot vs iterative fill
  - token budget

**Out of scope**
- Full retraining of frontier-scale models
- New MoE pretraining
- Large-scale RL
- Multi-node search-heavy methods

---

## **SMART Goal**
**Specific**  
Design and evaluate an inference-time method, **Cloze-of-Thought (CoT\*)**, that prompts an open 7B-8B LLM to generate a sparse reasoning scaffold with **15-40% of intermediate reasoning spans masked**, then iteratively fill the blanks to obtain a final answer.

**Measurable**  
Measure:
- **Final-answer accuracy** on GSM8K, MATH500, and BBH
- **Token cost** (generated reasoning tokens)
- **Error propagation rate** (cases where an early wrong step causes final failure)
- **Ablation gains** of uncertainty-based masking vs random masking

**Achievable**  
Use one open model such as **Llama-3.1-8B-Instruct** or **Qwen2.5-7B-Instruct**, with prompt-only inference and optional lightweight LoRA if needed. This is feasible on a **single GPU** using batch-limited evaluation.

**Relevant**  
Addresses a timely gap in inference-time reasoning: whether **partial omission plus iterative completion** is better than unrestricted left-to-right CoT.

**Time-bound**  
Within **8 weeks**, complete:
1. week 1-2: implement prompting framework and masking policies  
2. week 3-4: run benchmark evaluations  
3. week 5: ablations on masking ratio and fill order  
4. week 6: token-budget analysis and error taxonomy  
5. week 7: write paper draft  
6. week 8: finalize results and reproducibility package

### **Concrete SMART statement**
> **Within 8 weeks, demonstrate whether a Cloze-of-Thought inference strategy on a 7B-8B open LLM can achieve either (a) at least +2.0 absolute accuracy over standard CoT on at least 2 of 3 reasoning benchmarks under the same token budget, or (b) at least 25% fewer reasoning tokens at iso-accuracy, with statistically consistent gains over random masking and direct-answer baselines.**

---

## **Constraints**
- **Compute budget:** single GPU, ideally 24-48 GB VRAM; inference-only preferred
- **Training budget:** none or LoRA-only; no full finetuning required
- **Models/tools:** Hugging Face Transformers, vLLM, lm-eval-harness, Python, standard prompting pipelines
- **Data access:** public benchmarks only
- **Runtime target:** experiments should finish in **hours to low tens of hours**, not multi-day distributed runs
- **Evaluation constraint:** use deterministic decoding plus limited repeated runs for variance estimates

---

## **Benchmark**
### 1) **GSM8K**
- **Source:** Cobbe et al. grade-school math benchmark
- **Task:** multi-step arithmetic word problems
- **Metric:** exact-match answer accuracy
- **Why use it:** standard benchmark for testing whether reasoning format changes final correctness
- **Current SOTA:** yes; closed reasoning-oriented systems in 2024 report **mid-to-high 90% accuracy** on GSM8K

### 2) **MATH500**
- **Source:** 500-problem evaluation subset derived from the MATH benchmark
- **Task:** harder mathematical reasoning
- **Metric:** exact-match / answer accuracy
- **Why use it:** more sensitive than GSM8K to step-level reasoning failures
- **Current SOTA:** yes; strong 2024 reasoning systems report **roughly 85-90%+** depending on setup/model

### 3) **BBH (Big-Bench Hard)**
- **Source:** BIG-bench Hard reasoning subset
- **Task:** logical, symbolic, and compositional reasoning
- **Metric:** task accuracy averaged across subsets
- **Why use it:** tests whether any gains generalize beyond arithmetic
- **Current SOTA:** yes; top proprietary systems in 2024 report **around 90%+ average** on many BBH settings

### **Benchmark choice rationale**
These benchmarks are appropriate because:
- they are standard,
- they have strong baselines,
- they are small enough for a single-GPU inference study,
- and they directly test whether a new reasoning format improves final answer quality.

---

## **Success Criteria**
This becomes publishable if the study shows **more than a prompt gimmick**:

### **Minimum publishable outcome**
- CoT\* beats **standard CoT** by **>= 2 absolute points** on at least **two benchmarks**, **or**
- CoT\* matches CoT accuracy with **>= 25% fewer reasoning tokens**, **and**
- uncertainty-based masking outperforms random masking by a clear margin

### **Stronger publishable outcome**
- gains are largest on tasks with long reasoning chains,
- partial masking reduces variance across decoding seeds,
- masked-cloze reasoning is especially helpful when the model is prone to verbose but incorrect rationales,
- the paper includes an interpretable finding such as:
  - optimal masking ratio lies in a mid-range,
  - masking arithmetic transformations helps more than masking conclusions,
  - or iterative fill beats one-shot fill because it localizes uncertainty

### **Negative-but-still-valuable outcome**
Even if accuracy does not improve, the project is still useful if it shows:
- when partial masking harms reasoning,
- which kinds of blanks are recoverable vs brittle,
- and whether cloze-style reasoning is mainly a **token-efficiency** method rather than an **accuracy** method.

That would still be a meaningful contribution because this specific question is currently underexplored.

---

## **Generated**
**2025-02-14T00:00:00Z**