"""Inference strategies for CLOX experiments (v2).

Fixed implementations with proper:
- Token-level entropy from full softmax
- Real backward cloze reconstruction
- Compute-fair budgeting
- Distinct masking behavior for targeted vs random
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine import GenerationOutput, VLLMEngine, extract_answer, split_into_steps
from topology_v2 import (
    TopologyProfile,
    TraceAnalysis,
    analyze_trace,
    batch_estimate_topology,
    estimate_topology,
)


@dataclass
class StrategyResult:
    prediction: str
    confidence: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    strategy_name: str
    reasoning_trace: str
    step_metadata: list[dict[str, Any]] = field(default_factory=list)
    topology: TopologyProfile | None = None


class InferenceStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def run(
        self,
        engine: VLLMEngine,
        question: str,
        max_tokens: int = 512,
        few_shot: str = "",
    ) -> StrategyResult:
        ...


class StandardCoT(InferenceStrategy):
    name = "standard_cot"

    def run(self, engine, question, max_tokens=512, few_shot=""):
        prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."
        out = engine.generate_single(prompt, max_tokens=max_tokens)
        answer = extract_answer(out.text)
        confidence = math.exp(sum(out.logprobs) / max(len(out.logprobs), 1)) if out.logprobs else 0.0
        return StrategyResult(
            prediction=answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=out.total_tokens,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            strategy_name=self.name,
            reasoning_trace=out.text,
            step_metadata=[{"steps": split_into_steps(out.text)}],
        )


class SelfConsistency(InferenceStrategy):
    name = "self_consistency"

    def __init__(self, k: int = 8, temperature: float = 0.7):
        self.k = k
        self.temperature = temperature

    def run(self, engine, question, max_tokens=512, few_shot=""):
        prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."
        outputs = engine.generate_multi(
            prompt, n=self.k, max_tokens=max_tokens, temperature=self.temperature,
        )
        answers = [extract_answer(o.text) for o in outputs]
        vote_counts = Counter(answers)
        majority, count = vote_counts.most_common(1)[0]
        agreement = count / self.k

        total_prompt = sum(o.prompt_tokens for o in outputs)
        total_comp = sum(o.completion_tokens for o in outputs)
        return StrategyResult(
            prediction=majority,
            confidence=agreement,
            total_tokens=total_prompt + total_comp,
            prompt_tokens=total_prompt,
            completion_tokens=total_comp,
            strategy_name=self.name,
            reasoning_trace=outputs[0].text,
            step_metadata=[{
                "samples": len(outputs),
                "answers": answers,
                "votes": dict(vote_counts),
            }],
        )


class UncertaintyTargetedRepair(InferenceStrategy):
    """Entropy-guided selective step repair.

    1. Generate greedy CoT with full logprobs
    2. Compute per-step entropy from top-K softmax distribution
    3. Mask the highest-entropy steps
    4. Prompt model to repair masked steps
    """
    name = "targeted_repair"

    def __init__(self, mask_fraction: float = 0.4):
        self.mask_fraction = mask_fraction

    def run(self, engine, question, max_tokens=512, few_shot=""):
        prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."

        # Phase 1: Generate initial CoT with logprobs
        first = engine.generate_single(prompt, max_tokens=max_tokens, logprobs=20)
        trace = analyze_trace(first)

        if len(trace.steps) < 2:
            answer = extract_answer(first.text)
            return StrategyResult(
                prediction=answer, confidence=0.5,
                total_tokens=first.total_tokens,
                prompt_tokens=first.prompt_tokens,
                completion_tokens=first.completion_tokens,
                strategy_name=self.name, reasoning_trace=first.text,
            )

        # Phase 2: Identify high-entropy steps (REAL entropy from softmax)
        step_ents = trace.step_entropies
        n_mask = max(1, int(len(trace.steps) * self.mask_fraction))
        ranked = sorted(range(len(step_ents)), key=lambda i: step_ents[i], reverse=True)
        masked_indices = sorted(ranked[:n_mask])

        # Phase 3: Build repair prompt with masked steps
        visible = []
        for i, step in enumerate(trace.steps):
            if i in masked_indices:
                ent = step_ents[i] if i < len(step_ents) else 0.0
                visible.append(f"Step {i+1}: [HIGH UNCERTAINTY (entropy={ent:.2f}) - NEEDS CORRECTION]")
            else:
                visible.append(f"Step {i+1}: {step}")

        repair_prompt = (
            f"Question: {question}\n\n"
            f"A previous solution attempt found some uncertain steps. "
            f"The steps marked with [HIGH UNCERTAINTY] likely contain errors.\n\n"
            + "\n".join(visible)
            + "\n\nRewrite the complete solution, correcting the uncertain steps. "
            f"End with: The answer is <your answer>.\n"
        )

        # Phase 4: Generate repair with sampling (different from greedy first pass)
        repair = engine.generate_single(repair_prompt, max_tokens=max_tokens, temperature=0.3)
        answer = extract_answer(repair.text) or extract_answer(first.text)

        total_p = first.prompt_tokens + repair.prompt_tokens
        total_c = first.completion_tokens + repair.completion_tokens
        all_lps = first.logprobs + repair.logprobs
        confidence = math.exp(sum(all_lps) / max(len(all_lps), 1)) if all_lps else 0.0

        return StrategyResult(
            prediction=answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=total_p + total_c,
            prompt_tokens=total_p,
            completion_tokens=total_c,
            strategy_name=self.name,
            reasoning_trace=f"FIRST PASS:\n{first.text}\n\nREPAIR:\n{repair.text}",
            step_metadata=[{
                "masked_indices": masked_indices,
                "step_entropies": step_ents,
                "mask_reason": "entropy_ranked",
            }],
        )


class RandomRepair(InferenceStrategy):
    """Random step masking repair (ablation control for targeted repair)."""
    name = "random_repair"

    def __init__(self, mask_fraction: float = 0.4):
        self.mask_fraction = mask_fraction

    def run(self, engine, question, max_tokens=512, few_shot=""):
        prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."

        # Phase 1: Same greedy first pass
        first = engine.generate_single(prompt, max_tokens=max_tokens, logprobs=20)
        trace = analyze_trace(first)

        if len(trace.steps) < 2:
            answer = extract_answer(first.text)
            return StrategyResult(
                prediction=answer, confidence=0.5,
                total_tokens=first.total_tokens,
                prompt_tokens=first.prompt_tokens,
                completion_tokens=first.completion_tokens,
                strategy_name=self.name, reasoning_trace=first.text,
            )

        # Phase 2: Random masking (seeded by question hash + distinct salt)
        n_mask = max(1, int(len(trace.steps) * self.mask_fraction))
        rng = np.random.default_rng(hash(question) % (2**32) + 12345)
        masked_indices = sorted(
            rng.choice(len(trace.steps), size=min(n_mask, len(trace.steps)), replace=False).tolist()
        )

        visible = []
        for i, step in enumerate(trace.steps):
            if i in masked_indices:
                visible.append(f"Step {i+1}: [MASKED - fill in this step]")
            else:
                visible.append(f"Step {i+1}: {step}")

        repair_prompt = (
            f"Question: {question}\n\n"
            f"A previous solution has some steps masked. Fill in the missing steps:\n\n"
            + "\n".join(visible)
            + "\n\nComplete the solution by filling in all masked steps. "
            f"End with: The answer is <your answer>.\n"
        )

        repair = engine.generate_single(repair_prompt, max_tokens=max_tokens, temperature=0.3)
        answer = extract_answer(repair.text) or extract_answer(first.text)

        total_p = first.prompt_tokens + repair.prompt_tokens
        total_c = first.completion_tokens + repair.completion_tokens
        all_lps = first.logprobs + repair.logprobs
        confidence = math.exp(sum(all_lps) / max(len(all_lps), 1)) if all_lps else 0.0

        return StrategyResult(
            prediction=answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=total_p + total_c,
            prompt_tokens=total_p,
            completion_tokens=total_c,
            strategy_name=self.name,
            reasoning_trace=f"FIRST PASS:\n{first.text}\n\nREPAIR:\n{repair.text}",
            step_metadata=[{
                "masked_indices": masked_indices,
                "mask_reason": "random",
            }],
        )


class BackwardCloze(InferenceStrategy):
    """Answer-anchored backward cloze reconstruction.

    FIXED: Actually generates backward from answer to premises.

    1. Generate forward CoT to get candidate answer
    2. Anchor on answer and prompt backward verification:
       "Given the answer is X, reconstruct the reasoning from conclusion to premises"
    3. Check consistency between forward and backward chains
    """
    name = "backward_cloze"

    def __init__(self, n_forward: int = 3):
        self.n_forward = n_forward

    def run(self, engine, question, max_tokens=512, few_shot=""):
        prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."

        # Phase 1: Forward candidates
        forwards = engine.generate_multi(
            prompt, n=self.n_forward, max_tokens=max_tokens, temperature=0.7,
        )
        forward_answers = [extract_answer(o.text) for o in forwards]
        answer_counts = Counter(forward_answers)
        anchor_answer = answer_counts.most_common(1)[0][0] if answer_counts else forward_answers[0]

        # Phase 2: BACKWARD reconstruction (actually backward this time)
        backward_prompt = (
            f"Question: {question}\n\n"
            f"The answer to this question is: {anchor_answer}\n\n"
            f"Working BACKWARD from this answer, verify it is correct by "
            f"reconstructing the reasoning chain from the conclusion to the premises.\n\n"
            f"Start from the final result and trace back each step:\n"
            f"Final result: {anchor_answer}\n"
            f"Previous step: [derive what leads to this]\n"
            f"...\n"
            f"First step: [what we start from]\n\n"
            f"After backward verification, state whether the answer {anchor_answer} is correct. "
            f"If not, provide the correct answer.\n"
            f"End with: The answer is <your answer>.\n"
        )

        backward = engine.generate_single(backward_prompt, max_tokens=max_tokens)
        backward_answer = extract_answer(backward.text)

        # Phase 3: Consistency check
        if backward_answer == anchor_answer:
            final_answer = anchor_answer
            confidence = 0.9
        else:
            # Disagreement: try a tie-breaking forward pass
            tiebreak = engine.generate_single(prompt, max_tokens=max_tokens)
            tiebreak_answer = extract_answer(tiebreak.text)
            all_answers = forward_answers + [backward_answer, tiebreak_answer]
            final_counts = Counter(all_answers)
            final_answer = final_counts.most_common(1)[0][0]
            confidence = final_counts.most_common(1)[0][1] / len(all_answers)

        total_p = sum(o.prompt_tokens for o in forwards) + backward.prompt_tokens
        total_c = sum(o.completion_tokens for o in forwards) + backward.completion_tokens

        return StrategyResult(
            prediction=final_answer,
            confidence=confidence,
            total_tokens=total_p + total_c,
            prompt_tokens=total_p,
            completion_tokens=total_c,
            strategy_name=self.name,
            reasoning_trace=(
                f"FORWARD:\n{forwards[0].text}\n\n"
                f"BACKWARD:\n{backward.text}"
            ),
            step_metadata=[{
                "forward_answers": forward_answers,
                "anchor": anchor_answer,
                "backward_answer": backward_answer,
                "final": final_answer,
            }],
        )


class FullRegeneration(InferenceStrategy):
    """Critique + full rewrite."""
    name = "full_regeneration"

    def run(self, engine, question, max_tokens=512, few_shot=""):
        prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."
        first = engine.generate_single(prompt, max_tokens=max_tokens)
        first_answer = extract_answer(first.text)

        critique_prompt = (
            f"Question: {question}\n\n"
            f"A student wrote this solution:\n{first.text}\n\n"
            f"Review for errors and write a corrected, complete solution. "
            f"End with: The answer is <your answer>.\n"
        )
        second = engine.generate_single(critique_prompt, max_tokens=max_tokens, temperature=0.3)
        answer = extract_answer(second.text) or first_answer

        total_p = first.prompt_tokens + second.prompt_tokens
        total_c = first.completion_tokens + second.completion_tokens
        all_lps = first.logprobs + second.logprobs
        confidence = math.exp(sum(all_lps) / max(len(all_lps), 1)) if all_lps else 0.0

        return StrategyResult(
            prediction=answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=total_p + total_c,
            prompt_tokens=total_p,
            completion_tokens=total_c,
            strategy_name=self.name,
            reasoning_trace=f"FIRST:\n{first.text}\n\nREVISED:\n{second.text}",
        )


class HierarchicalRepair(InferenceStrategy):
    """Bottleneck-aware hierarchical repair.

    Identifies steps that are both uncertain AND have many downstream dependents.
    Prioritizes repairing these "bottleneck" steps first.
    """
    name = "hierarchical_repair"

    def __init__(self, mask_fraction: float = 0.3):
        self.mask_fraction = mask_fraction

    def run(self, engine, question, max_tokens=512, few_shot=""):
        prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."
        first = engine.generate_single(prompt, max_tokens=max_tokens, logprobs=20)
        trace = analyze_trace(first)

        if len(trace.steps) < 3:
            answer = extract_answer(first.text)
            return StrategyResult(
                prediction=answer, confidence=0.5,
                total_tokens=first.total_tokens,
                prompt_tokens=first.prompt_tokens,
                completion_tokens=first.completion_tokens,
                strategy_name=self.name, reasoning_trace=first.text,
            )

        # Bottleneck score = entropy × downstream_weight
        step_scores = []
        for i in range(len(trace.steps)):
            ent = trace.step_entropies[i] if i < len(trace.step_entropies) else 0.0
            downstream = (len(trace.steps) - i - 1) / max(len(trace.steps) - 1, 1)
            score = ent * (1.0 + downstream)
            step_scores.append((i, score))

        step_scores.sort(key=lambda x: x[1], reverse=True)
        n_mask = max(1, int(len(trace.steps) * self.mask_fraction))
        masked_indices = sorted([idx for idx, _ in step_scores[:n_mask]])

        visible = []
        for i, step in enumerate(trace.steps):
            if i in masked_indices:
                score = next(s for idx, s in step_scores if idx == i)
                visible.append(f"Step {i+1}: [BOTTLENECK (score={score:.2f}) - VERIFY AND CORRECT]")
            else:
                visible.append(f"Step {i+1}: {step}")

        repair_prompt = (
            f"Question: {question}\n\n"
            f"Critical steps in this solution have been flagged as potential bottlenecks "
            f"(high uncertainty + many dependent steps). Verify and correct them:\n\n"
            + "\n".join(visible)
            + "\n\nCarefully verify each flagged step and rewrite the complete solution. "
            f"End with: The answer is <your answer>.\n"
        )

        repair = engine.generate_single(repair_prompt, max_tokens=max_tokens, temperature=0.3)
        answer = extract_answer(repair.text) or extract_answer(first.text)

        total_p = first.prompt_tokens + repair.prompt_tokens
        total_c = first.completion_tokens + repair.completion_tokens

        return StrategyResult(
            prediction=answer,
            confidence=0.6,
            total_tokens=total_p + total_c,
            prompt_tokens=total_p,
            completion_tokens=total_c,
            strategy_name=self.name,
            reasoning_trace=f"FIRST:\n{first.text}\n\nHIERARCHICAL REPAIR:\n{repair.text}",
            step_metadata=[{
                "bottleneck_indices": masked_indices,
                "scores": step_scores,
            }],
        )


class CLOXAdaptive(InferenceStrategy):
    """Topology-aware adaptive strategy selector.

    1. Run pilot traces to estimate topology (r̄, ℓ)
    2. Select optimal strategy based on Theorems 1-3
    3. Execute selected strategy with remaining budget
    """
    name = "clox_adaptive"

    def __init__(self, n_pilot: int = 5, sc_k: int = 8):
        self.n_pilot = n_pilot
        self.sc_k = sc_k
        self.strategies = {
            "standard_cot": StandardCoT(),
            "self_consistency": SelfConsistency(k=sc_k),
            "targeted_repair": UncertaintyTargetedRepair(),
            "hierarchical_repair": HierarchicalRepair(),
        }

    def run(self, engine, question, max_tokens=512, few_shot=""):
        # Phase 1: Topology estimation (uses pilot budget)
        topo = estimate_topology(
            engine, question, few_shot=few_shot,
            n_pilot=self.n_pilot, max_tokens=max_tokens,
            do_regeneration_test=False,  # save compute for main strategy
        )

        selected = topo.strategy
        if selected == "adaptive":
            selected = "targeted_repair"

        strategy = self.strategies.get(selected, self.strategies["standard_cot"])

        # Phase 2: Run selected strategy
        result = strategy.run(engine, question, max_tokens=max_tokens, few_shot=few_shot)

        # Add topology metadata
        result.strategy_name = f"clox_adaptive({selected})"
        result.topology = topo
        # Account for pilot tokens (n_pilot samples × ~max_tokens each)
        pilot_tokens = self.n_pilot * max_tokens  # upper bound estimate
        result.total_tokens += pilot_tokens
        result.step_metadata.append({
            "topology": {
                "r_bar": topo.r_bar,
                "epl": topo.epl,
                "n_steps": topo.n_steps,
                "selected": selected,
            },
        })
        return result


class ComputeMatchedSC(InferenceStrategy):
    """Self-consistency with compute budget matched to 2-pass strategies.

    Uses the same total token budget as repair strategies (2× single pass)
    by adjusting the number of samples.
    """
    name = "compute_matched_sc"

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature

    def run(self, engine, question, max_tokens=512, few_shot=""):
        prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."
        # 2-pass strategies use ~2× tokens, so we use 2 samples
        # (matching the compute of a first pass + repair pass)
        outputs = engine.generate_multi(
            prompt, n=2, max_tokens=max_tokens, temperature=self.temperature,
        )
        answers = [extract_answer(o.text) for o in outputs]
        vote_counts = Counter(answers)
        majority, count = vote_counts.most_common(1)[0]

        total_p = sum(o.prompt_tokens for o in outputs)
        total_c = sum(o.completion_tokens for o in outputs)
        return StrategyResult(
            prediction=majority,
            confidence=count / len(outputs),
            total_tokens=total_p + total_c,
            prompt_tokens=total_p,
            completion_tokens=total_c,
            strategy_name=self.name,
            reasoning_trace=outputs[0].text,
            step_metadata=[{"answers": answers, "k": len(outputs)}],
        )


class BackwardAnchoredVerification(InferenceStrategy):
    """BAV: use backward cloze as a verifier for forward CoT.

    Pipeline:
      1. Greedy forward CoT → answer A (≈ 1 pass).
      2. Answer-anchored backward reconstruction on A → backward answer B.
      3. If A == B: commit to A.
      4. Else: SC fallback with k=fallback_k samples (majority vote over all).

    Key differences vs BackwardCloze:
      - Only 1 forward pass (not 3), so baseline cost is CoT + backward ≈ CMS budget.
      - Fallback is triggered ONLY on disagreement, yielding adaptive compute.
      - Agreement rate determines average token cost; if most problems agree,
        BAV stays cheap.
    """
    name = "bav"

    def __init__(self, fallback_k: int = 5, temperature: float = 0.7):
        self.fallback_k = fallback_k
        self.temperature = temperature

    def run(self, engine, question, max_tokens=512, few_shot=""):
        prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."

        # Phase 1: greedy forward CoT
        forward = engine.generate_single(prompt, max_tokens=max_tokens)
        forward_answer = extract_answer(forward.text)

        # Phase 2: answer-anchored backward verification
        backward_prompt = (
            f"Question: {question}\n\n"
            f"A candidate answer is: {forward_answer}\n\n"
            f"Verify by reasoning backward from this answer to the given information.\n"
            f"Starting from the conclusion ({forward_answer}), reconstruct each step "
            f"back to the original premises.\n\n"
            f"After backward verification, either confirm the answer or provide "
            f"the correct answer if the verification fails.\n"
            f"End with: The answer is <answer>."
        )
        backward = engine.generate_single(backward_prompt, max_tokens=max_tokens)
        backward_answer = extract_answer(backward.text)

        total_p = forward.prompt_tokens + backward.prompt_tokens
        total_c = forward.completion_tokens + backward.completion_tokens
        agreed = (forward_answer == backward_answer) and forward_answer != ""

        if agreed:
            # High confidence: commit to forward answer
            final_answer = forward_answer
            confidence = 0.9
            fallback_used = False
            fallback_samples = []
        else:
            # Disagreement: trigger SC fallback
            fallback = engine.generate_multi(
                prompt, n=self.fallback_k, max_tokens=max_tokens,
                temperature=self.temperature,
            )
            fallback_samples = [extract_answer(o.text) for o in fallback]
            # Include forward and backward in the vote
            all_answers = [forward_answer, backward_answer] + fallback_samples
            vote_counts = Counter([a for a in all_answers if a])
            if vote_counts:
                final_answer, count = vote_counts.most_common(1)[0]
                confidence = count / max(len(all_answers), 1)
            else:
                final_answer = forward_answer
                confidence = 0.3
            total_p += sum(o.prompt_tokens for o in fallback)
            total_c += sum(o.completion_tokens for o in fallback)
            fallback_used = True

        return StrategyResult(
            prediction=final_answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=total_p + total_c,
            prompt_tokens=total_p,
            completion_tokens=total_c,
            strategy_name=self.name,
            reasoning_trace=(
                f"FORWARD:\n{forward.text}\n\nBACKWARD:\n{backward.text}"
                + (f"\n\nFALLBACK ({self.fallback_k} samples)" if fallback_used else "")
            ),
            step_metadata=[{
                "forward_answer": forward_answer,
                "backward_answer": backward_answer,
                "agreed": agreed,
                "fallback_used": fallback_used,
                "fallback_samples": fallback_samples,
                "final": final_answer,
                "fallback_k": self.fallback_k,
            }],
        )


class SCK3(SelfConsistency):
    """SC with k=3 — matches BAV+fallback budget roughly."""
    name = "sc_k3"
    def __init__(self, temperature: float = 0.7):
        super().__init__(k=3, temperature=temperature)


class SCK5(SelfConsistency):
    """SC with k=5 — matches BAV+fallback budget at high disagreement rate."""
    name = "sc_k5"
    def __init__(self, temperature: float = 0.7):
        super().__init__(k=5, temperature=temperature)


STRATEGY_REGISTRY: dict[str, type[InferenceStrategy]] = {
    "standard_cot": StandardCoT,
    "self_consistency": SelfConsistency,
    "compute_matched_sc": ComputeMatchedSC,
    "sc_k3": SCK3,
    "sc_k5": SCK5,
    "targeted_repair": UncertaintyTargetedRepair,
    "random_repair": RandomRepair,
    "backward_cloze": BackwardCloze,
    "full_regeneration": FullRegeneration,
    "hierarchical_repair": HierarchicalRepair,
    "clox_adaptive": CLOXAdaptive,
    "bav": BackwardAnchoredVerification,
}


def build_strategy(name: str, **kwargs) -> InferenceStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    return cls(**kwargs)
