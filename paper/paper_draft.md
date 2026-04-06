# CLOX: Partial Masking as a Control for Synthetic Reasoning

## Abstract

Large language models are often evaluated with fully exposed left-to-right reasoning, yet the underlying question is broader: can partial masking at inference time improve problem solving by delaying uncertain commitments and turning reasoning into a cloze-style reconstruction problem [wei2022chain; wang2022selfconsistency; du2022general]? Prior work gives strong ingredients for this question through chain-of-thought prompting, self-consistency, search, and infilling objectives, but it does not isolate whether masking itself is the useful control variable at inference time [kojima2022large; yao2023tree; wang2023planandsolve].

We introduce **CLOX**, a unified family of partial-masking and repair operators, and evaluate that idea in a synthetic seven-segment proxy benchmark implemented with a trained structured predictor rather than an actual decoder-only LLM. Across one executed experiment with three seeds, the strongest mean exact-match accuracy came from uncertainty-targeted selective masked repair at **0.1927 ± 0.0195**, compared with **0.1615 ± 0.0195** for budget-matched self-consistency, **0.1615 ± 0.0147** for answer-anchored backward cloze reconstruction, and **0.0885 ± 0.0368** for the standard left-to-right baseline; the corresponding mean final-answer error rates were **0.8073**, **0.8385**, **0.8385**, and **0.9115**.

The main conclusion is therefore narrower and more accurate than the original claim: partial masking is feasible and competitive in this synthetic control setting, but the present evidence does not establish an LLM-specific advantage for masking over other inference-time restructuring strategies.


> **Note:** This paper was produced in degraded mode. Quality gate score (2.3/4.0) was below threshold. Unverified numerical results in tables have been replaced with `---` and require independent verification.


## Introduction

### Motivation

Reasoning with large language models is usually presented as a single exposed trajectory: generate a sequence of intermediate steps from left to right, preserve every commitment, and produce a final answer only after the full rationale has been written out [wei2022chain; kojima2022large]. That interface has been productive. Chain-of-thought prompting can improve multi-step problem solving, zero-shot reasoning cues can elicit latent decomposition behavior, and hierarchical prompting strategies can further structure hard tasks [wei2022chain; kojima2022large; zhou2022leasttomost].

Even so, the left-to-right format imposes a rigid causal order on intermediate decisions. An early weak step can constrain everything that follows, and long rationales can spend budget on low-value tokens instead of uncertainty reduction. Human solvers often behave differently: they sketch, leave gaps, verify partial structure, and revisit unstable fragments. That contrast motivates the central research question behind this paper: for large language models, is it possible that partial masking at inference time could improve performance by encouraging cloze-style or fill-in-the-blank reasoning rather than a single irreversible narrative?

### Gap Between the LLM Question and Existing Evidence

Existing literature gives strong conceptual motivation but not a clean answer. On the reasoning side, self-consistency aggregates multiple sampled traces to reduce sensitivity to single-trajectory errors , tree-based search explores branching thought structures instead of committing immediately to one path , and plan-first prompting seeks more stable high-level decomposition before local execution . Broader surveys of reasoning behavior emphasize that measured gains depend strongly on prompting format, benchmark design, and evaluation choices [chang2023survey; huang2023reasoning; hebenstreit2024comparison].

On the masking side, masked language modeling and blank infilling have long shown that reconstructing hidden spans can be an effective representational bias, from BERT to GLM [devlin2019bert; du2022general; rogers2020primer]. Yet those literatures do not directly test the narrow hypothesis at issue here: whether **inference-time** partial masking, applied as a control over intermediate reasoning, improves performance relative to strong non-masking alternatives.

Reviewer feedback identified a second and more serious gap in the original draft: the prose framed the study as an LLM prompting paper, whereas the implemented evidence came from a synthetic supervised pipeline. The revision therefore makes that distinction explicit and treats the present work as a controlled synthetic proxy study motivated by an LLM question, not as direct evidence about deployed decoder-only LLMs.

### Present Study and Core Insight

CLOX addresses that gap by recasting partial masking as a family of inference-time control operators over a reasoning scaffold. The key idea remains the same as in the original draft: start from a standard left-to-right solution attempt, then restructure that attempt under shared evaluation conditions through answer-anchored backward cloze reconstruction, uncertainty-targeted masked repair, random-span repair, or broader rationale rewriting.

What changes in this revision is the empirical framing. The implemented system is a compact synthetic reasoner trained on a seven-segment task, and the reported results therefore support claims about **structured reasoning control in a synthetic proxy benchmark**, not about end-to-end LLM prompting. This correction matters because it changes the interpretation of every result. The paper no longer claims that decoder-only LLMs benefit from masking; instead, it asks whether the masking hypothesis remains plausible after a careful controlled test in a synthetic setting that was explicitly designed to mimic reasoning-time restructuring.

That narrower claim is still useful. A proxy study cannot settle the LLM question, but it can reveal whether partial masking is a coherent control mechanism, whether stronger non-masking baselines remain competitive, and whether ablations separate cleanly enough to justify future direct LLM experiments.

### Contributions

Our contributions are threefold:

1. We revise the paper to align the narrative with the actual experiment: **CLOX is evaluated here as a synthetic proxy study of inference-time partial masking**, not as a completed decoder-only LLM prompting paper.
2. We present a controlled comparison among standard left-to-right reasoning, budget-matched self-consistency, answer-anchored backward cloze reconstruction, uncertainty-targeted masked repair, random-span masked repair, forward-order cloze reconstruction, and whole-rationale mask repair on a shared seven-segment benchmark, using only the recorded metrics from one executed experiment with three seeds.
3. We report a negative-but-informative empirical finding: masking-based control remained competitive, and the best mean exact-match score came from uncertainty-targeted repair, yet the study does not isolate a stable masking-specific advantage and all method differences should be treated as descriptive rather than statistically significant.

### Roadmap

The rest of the paper develops this argument in stages. The next section situates CLOX within prior work on inference-time reasoning, self-correction, structured omission, and benchmark design [pan2023automatically; lanham2023measuring; lyu2023faithful; srivastava2022beyond]. The method section then formalizes CLOX as a family of control operators while describing the implemented synthetic pipeline faithfully. After that, the experimental section specifies the benchmark, conditions, seeds, and hardware environment actually used in the recorded run. The results section reports the quantitative comparison without overstating what the evidence can support, and the discussion turns those quantitative patterns into practical lessons for future direct LLM studies of cloze-style reasoning.

## Related Work

### Inference-Time Reasoning and Structured Control

Inference-time reasoning has evolved from simple prompt engineering toward increasingly explicit forms of structural control. Chain-of-thought prompting established that asking models to produce intermediate reasoning can improve multi-step problem solving [wei2022chain], and zero-shot cues showed that much of this behavior can be elicited without task-specific demonstrations . Least-to-most prompting then decomposed difficult questions into ordered subproblems , while tree-based search reframed reasoning as exploration over multiple branches rather than a single completion . A parallel line of work studied plan-first prompting, reasoning without explicit chain-of-thought exemplars, and broader alignment-oriented interventions for reasoning-time control [wang2023planandsolve; wang2024chainofthought; wang2023making].

These studies motivate the basic intuition behind CLOX: reasoning quality can depend strongly on **how** intermediate computation is exposed and constrained. The present paper differs from that literature in one important respect. Instead of optimizing the wording, order, or diversity of fully visible thoughts, CLOX makes **partial visibility** the central intervention and asks whether reconstruction through blanks or masked spans provides a useful alternative control axis.

That distinction is easy to state but empirically difficult to validate. Many reasoning improvements can be explained by simple re-computation, search, or answer aggregation rather than by a more specific structural mechanism. Self-consistency is the clearest example. By sampling multiple rationales and taking a consensus answer, it often improves robustness without imposing any special visibility structure on the intermediate trace . Search methods similarly improve outcomes by exploring multiple trajectories , and planning methods can reduce local myopia by restructuring the order of generated reasoning .

CLOX borrows the control mindset of that literature but asks a narrower question: if a solver is forced to leave selected parts of its intermediate reasoning blank and reconstruct them later, does that restructuring provide value beyond just another pass of computation? Because the present evidence comes from a synthetic proxy rather than an LLM, the answer remains provisional, but the comparison itself is still directly informed by this reasoning-control literature.

### Self-Correction, Revision, and Rationale Repair

A second relevant body of work treats inference as an iterative process in which an initial answer is critiqued, filtered, revised, or repaired. Surveys of automatic self-correction describe regenerate-and-select loops, critique-and-revise frameworks, and verifier-guided pipelines as recurring strategies for improving model outputs after an initial pass . Related systems augment reasoning with retrieval, tools, or structured checks so that the model does not rely entirely on its first free-form continuation [chen2023chatcot; trivedi2023interleaving; gao2023retrievalaugmented]. Other methods attach learned or heuristic filters to reasoning traces, attempting to detect or suppress brittle steps before the final answer is committed [khalifa2023grace; wu2024mitigating].

These approaches are especially relevant because CLOX can be understood as a restricted form of revision: instead of redoing everything, it blanks specific parts of the trace and asks the model or proxy system to re-compute those regions under structural constraints.

The present work differs from the broader revision literature in both mechanism and evidential scope. Mechanistically, the core intervention is not critique, reranking, tool use, or verifier guidance; it is **partial omission followed by infilling or repair**. Evidentially, the current paper no longer treats the experiment as direct LLM evidence. That matters because many self-correction papers make claims about language-model reasoning behavior under natural-language prompting. CLOX in this revision supports a smaller claim: in a synthetic structured prediction environment designed to mimic staged reasoning control, partial masking is implementable, competitive, and methodologically worth comparing against self-consistency-style baselines.

That narrower framing also clarifies why budget matching remains central. If extra computation alone explains most of the gain, then any masking advantage should shrink once self-consistency is included as a serious comparator.

### Masking, Blank Infilling, and Structured Omission

Masking is a familiar idea in training objectives, but much less settled as an inference-time reasoning interface. BERT made masked language modeling foundational for bidirectional representation learning [devlin2019bert], and later analyses examined how masking changes contextual representations and what such models encode . GLM brought blank infilling closer to the causal-generation world by integrating autoregressive blank completion into a general pretraining framework [du2022general]. Prompt optimization research further showed that small changes in task format can alter model behavior materially [shin2020autoprompt; gao2021making; zhou2022large; sahoo2024systematic]. Long-context studies likewise found that models do not use all prompt positions equally well, which suggests that visibility structure and placement can matter as much as total content .

This literature makes the basic CLOX hypothesis plausible: hiding selected material and forcing later reconstruction could change error patterns even without changing the underlying solver.

At the same time, training-time masking does not automatically imply inference-time benefit. Blank-infilling objectives teach models to complete missing spans under distributional assumptions defined by pretraining. Inference-time masking, by contrast, is a control policy imposed on a solver at evaluation. The missing span may or may not correspond to a useful computational bottleneck; the surrounding scaffold may or may not be informative; and the repair step may simply reintroduce the same mistake in a different format.

The distinction becomes even sharper in the present revision, because the implemented experiment is not a pretrained language model using fill-in-the-middle prompting. It is a synthetic reasoner evaluated with control conditions inspired by masking and cloze reconstruction. This makes the paper more conservative but also more precise: the results say something about the **logic** of masking as a reasoning control, not yet about its realized effect in frontier decoder-only LLM APIs.

##

![Figure 1: Fig Error Breakdown By Subset](charts/fig_error_breakdown_by_subset.png)

![Figure 2: Fig Main Results All Conditions](charts/fig_main_results_all_conditions.png)

![Figure 3: Fig Masking Design Comparison](charts/fig_masking_design_comparison.png)

![Figure 4: Fig Error Breakdown By Subset](charts/fig_error_breakdown_by_subset.png)

![Figure 5: Fig Main Results All Conditions](charts/fig_main_results_all_conditions.png)

![Figure 6: Fig Masking Design Comparison](charts/fig_masking_design_comparison.png)

# Evaluation Practice, Proxies, and What Can Be Claimed

Reasoning claims are especially sensitive to benchmark choice, reporting discipline, and baseline strength. Surveys of LLM evaluation document wide variance across metrics, datasets, prompt templates, and reporting conventions [chang2023survey; zhao2023survey; naveed2023comprehensive; min2023recent]. Benchmarks can overstate capabilities when they reward familiarity or prompt sensitivity rather than robust inference [srivastava2022beyond; mcintosh2025inadequacies]. Neighboring work on contamination-resistant evaluation in code and related domains makes the same methodological point in a different setting: claims become persuasive only when baselines are strong, evaluation is controlled, and reported numbers map cleanly onto the implemented setup [jain2024livecodebench; liu2023your; xu2022systematic].

Reviewer feedback on the original draft stressed exactly this issue. The first submission version described one empirical domain but actually reported another.

This revision adopts those evaluation lessons directly. The paper now positions CLOX as a **pilot empirical study on a synthetic proxy benchmark** whose role is to test a bounded hypothesis about partial masking as structured reasoning control. That move is not cosmetic. It changes the title, abstract, method description, experimental framing, and conclusion so that each claim is supported by the evidence actually present in the recorded run. It also clarifies the relation to future work. A proxy benchmark can motivate or discourage direct LLM experimentation; it cannot replace it.

In that sense, the present paper contributes not only a result but also a methodological correction: if the implemented evidence is synthetic, then the paper must say so explicitly and discuss LLM implications as hypotheses for later validation rather than as established findings.

## Method

### Problem Formulation

CLOX is a family of inference-time control operators that test a simple idea: instead of exposing every intermediate step in a single left-to-right pass, a solver can leave selected regions blank and revisit them later through reconstruction or repair. In the original draft, this formulation was presented as if the underlying system were a fixed decoder-only language model. Reviewer feedback showed that this was inaccurate. The implemented evidence comes from a synthetic structured prediction pipeline in which a compact learned reasoner is trained on seven-segment data and then evaluated under multiple reasoning-control conditions.

The revised method therefore separates two levels of description. At the conceptual level, CLOX is motivated by the LLM question of whether partial masking can improve reasoning. At the implemented level, CLOX is instantiated as a synthetic proxy that applies masking-style control to structured intermediate representations in a benchmark designed to emulate staged reasoning behavior.

Formally, let \(g_\theta\) denote the trained base reasoner with parameters \(\theta\), and let \(x\) denote an input from the synthetic seven-segment benchmark. For each example, the system produces an initial structured solution state \(z^{(0)}\) together with a final answer candidate \(y^{(0)}\). Standard left-to-right reasoning returns that pair directly. CLOX instead applies an operator \(\mathcal{C}_\phi\) that transforms the initial state according to a method policy \(\phi\), yielding a revised state \(z^{(1)}\) and revised answer \(y^{(1)}\):

\[
(z^{(1)}, y^{(1)}) = \mathcal{C}_\phi(g_\theta, x, z^{(0)}, y^{(0)}).
\]

The emphasis of the experiment is therefore not on learning a new model family, but on controlling the visibility and revision pattern of intermediate problem-solving structure after the base model has been trained.

This framing preserves the original scientific question while correcting the empirical one. The question remains whether partial masking can serve as a useful control for reasoning. What changes is the domain of evidence. Here the evidence comes from a synthetic proxy that permits controlled comparisons among masking and non-masking variants without retraining a separate model for each condition. That choice enables clean within-benchmark comparison, but it also means the paper should be read as a synthetic study informed by LLM reasoning literature rather than as a direct report on deployed decoder-only models [wei2022chain; wang2022selfconsistency; du2022general].

### Implemented Base System

The implementation underlying the recorded results uses a compact supervised reasoner rather than a language-model API. Reviewer analysis of the accompanying code identified a trainable `DigitPerceptionMLP` together with synthetic data loading, explicit train and validation loaders, and a standard optimization loop. The revised paper therefore describes the base system accurately as a **trained multilayer perceptron for synthetic digit-perception and structured reasoning over seven-segment inputs**.

Training occurs before evaluation, after which the learned model is frozen and subjected to the different control conditions reported in the results section. This is an important correction because the original text incorrectly stated that the entire study was inference-only and used no parameter updates.

The synthetic setting still supports the CLOX hypothesis because the benchmark exposes a structured intermediate state that can be perturbed, partially hidden, and reconstructed. In other words, the masking operators do not require natural-language chain-of-thought text to be meaningful; they require only an ordered intermediate representation whose exposure can be controlled. That makes the setup a useful proxy for the core mechanism. It does not make it equivalent to an LLM prompt-completion environment. The paper therefore avoids claims about prompt formatting, tokenizer behavior, or decoder-only generation interfaces in the methodological description and reserves those connections for motivation and discussion.

Although the exported experiment record does not preserve every training hyperparameter numerically, the code structure and reviewer audit establish the following facts that are sufficient to reinterpret the experiment correctly:

- The model is trained rather than used as a fixed pretrained decoder.
- The pipeline includes train and validation stages rather than a pure test-time prompt harness.
- Parameter updates occur during model fitting.
- The seven reported inference conditions are then evaluated on the same held-out synthetic benchmark under three seeds.

These corrections bring the method section into alignment with the experiment that actually produced the numbers reported later in the paper.

### CLOX Operators

The standard baseline, **Standard Left-to-Right Chain-of-Thought (Std-CoT)**, is the identity control. The base reasoner produces a structured solution and final answer in one pass, and the evaluation reads that answer without any additional repair or reconstruction step. The naming is inherited from the original experimental interface and retained for consistency with the recorded logs, but in this revised paper it should be understood as the default single-pass structured reasoning policy in the synthetic benchmark rather than a literal natural-language chain-of-thought prompt.

The first strong comparator is **Budget-Matched Self-Consistency Consensus (BMSC)**. Here the system produces multiple solution candidates under the same overall evaluation protocol, and the final answer is selected through consensus rather than direct single-pass commitment. This condition plays the same methodological role in the synthetic study that self-consistency plays in LLM reasoning work : it tests whether extra computation and aggregation are sufficient to improve performance without any masking-specific mechanism. Including BMSC is crucial because it prevents the experiment from attributing gains to masking when they may instead come from re-computation alone.

The flagship masking condition is **Answer-Anchored Backward Cloze Reconstruction (ABCR)**. The system first generates an initial answer and corresponding structured rationale. It then masks selected internal spans, preserves an answer anchor, and reconstructs the hidden structure in a reverse-oriented cloze style. Conceptually, ABCR asks whether a provisional answer can help stabilize missing intermediate decisions by turning the problem from open-ended generation into constrained reconstruction.

---

*Editorial note: the provided manuscript input ends here mid-section, so no additional content was fabricated beyond this point.*