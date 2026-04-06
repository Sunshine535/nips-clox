# Papers: CLOX (Must-Cite Core)

## Core Chain-of-Thought and Reasoning

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., NeurIPS 2022)
   - https://arxiv.org/abs/2201.11903
   - Primary baseline; all claims are relative to standard free-form CoT.

2. **Large Language Models are Zero-Shot Reasoners** (Kojima et al., NeurIPS 2022)
   - https://arxiv.org/abs/2205.11916
   - Establishes zero-shot CoT as a baseline prompting strategy.

3. **Self-Consistency Improves Chain of Thought Reasoning in Language Models** (Wang et al., ICLR 2023)
   - https://arxiv.org/abs/2203.11171
   - Key compute-matched baseline; represents the standard extra-compute reasoning approach.

4. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** (Yao et al., NeurIPS 2023)
   - https://arxiv.org/abs/2305.10601
   - Search-based reasoning at inference time; context for structured reasoning beyond L2R.

5. **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models** (Zhou et al., ICLR 2023)
   - https://arxiv.org/abs/2205.10625
   - Decomposition-based prompting; relevant to structured reasoning ordering.

## Masked Language Modeling and Blank Infilling

6. **GLM: General Language Model Pretraining with Autoregressive Blank Infilling** (Du et al., ACL 2022)
   - https://doi.org/10.18653/v1/2022.acl-long.26
   - Most directly relevant architecture: autoregressive blank infilling in a general LM.

7. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (Devlin et al., NAACL 2019)
   - Foundational masked language model; conceptual ancestor of cloze-style inference.

8. **A Primer in BERTology: What We Know About How BERT Works** (Rogers et al., TACL 2020)
   - https://doi.org/10.1162/tacl_a_00349
   - Survey of masked modeling behavior; useful for understanding cloze mechanisms.

## Self-Correction and Revision

9. **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning** (Wang et al., ACL 2023)
   - https://doi.org/10.18653/v1/2023.acl-long.147
   - Structured prompting for reasoning; related to scaffold-then-fill approach.

10. **Automatically Correcting Large Language Models: Surveying Self-Correction Strategies** (Pan et al., 2023)
    - https://arxiv.org/abs/2308.03188
    - Survey of self-correction; context for repair-based methods.

## Context and Attention

11. **Lost in the Middle: How Language Models Use Long Contexts** (Liu et al., TACL 2024)
    - https://doi.org/10.1162/tacl_a_00638
    - Relevant to understanding how models attend to partially masked reasoning traces.

## Curation Rule

- Only keep papers directly needed by the CLOX claim or essential for baseline/method justification.
- Verify venue metadata before adding newer items.
- Separate foundational references from recent concurrent work.
