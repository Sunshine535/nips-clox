from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from topology import (
    ReasoningStep,
    ReasoningTrace,
    TopologyEstimator,
    TopologyProfile,
    _split_rationale_into_steps,
)


@dataclass
class StrategyResult:
    prediction: str
    confidence: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    logprobs: list[float] = field(default_factory=list)
    step_metadata: list[dict[str, Any]] = field(default_factory=list)
    strategy_name: str = ""
    reasoning_trace: str = ""
    pilot_traces: list[ReasoningTrace] = field(default_factory=list)
    topology: TopologyProfile | None = None


def _extract_answer(text: str) -> str:
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"####\s*(.+)",
        r"(?:final answer|the answer is|answer:)\s*[:\s]*([^\n.;]+)",
        r"(?:Therefore|So|Thus|Hence),?\s*(?:the (?:final )?answer is)?\s*([^\n.;]+)",
    ]
    # Search from end of text backward for the LAST match (most likely the final answer)
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            ans = matches[-1].group(1).strip()
            # Filter out clearly non-answer extractions
            if len(ans) > 0 and len(ans) < 200:
                return ans
    # Fallback: last non-empty line that looks like an answer
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        # Skip lines that are clearly meta-text
        if any(skip in line.lower() for skip in ["let's", "fill in", "step", "masked", "blank", "repair"]):
            continue
        # Look for numeric answers or short answers
        num = re.search(r"[-−]?\d+(?:\.\d+)?", line)
        if num:
            return num.group()
        if len(line) < 50:
            return line
    return lines[-1].strip() if lines else ""


def _generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    return_logprobs: bool = True,
) -> tuple[str, list[float], int]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": return_logprobs,
    }
    if do_sample and temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

    generated_ids = outputs.sequences[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_tokens = len(generated_ids)

    logprobs = []
    if return_logprobs and hasattr(outputs, "scores") and outputs.scores:
        for step_idx, score in enumerate(outputs.scores):
            if step_idx >= len(generated_ids):
                break
            log_probs = F.log_softmax(score[0].float(), dim=-1)
            token_id = generated_ids[step_idx]
            logprobs.append(float(log_probs[token_id].cpu()))

    return generated_text, logprobs, completion_tokens


def _compute_token_entropy(logprobs: list[float], window: int = 5) -> list[float]:
    if not logprobs:
        return []
    entropies = []
    for i in range(len(logprobs)):
        start = max(0, i - window // 2)
        end = min(len(logprobs), i + window // 2 + 1)
        window_lps = logprobs[start:end]
        probs = [math.exp(lp) for lp in window_lps]
        mean_prob = sum(probs) / len(probs)
        entropy = -mean_prob * math.log(max(mean_prob, 1e-10))
        entropies.append(entropy)
    return entropies


class InferenceStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def run(
        self,
        model,
        tokenizer,
        question: str,
        max_new_tokens: int = 512,
        few_shot_prompt: str = "",
        token_budget: int | None = None,
    ) -> StrategyResult:
        ...


class StandardCoT(InferenceStrategy):
    name = "standard_cot"

    def __init__(self, few_shot: bool = True):
        self.few_shot = few_shot

    def run(self, model, tokenizer, question, max_new_tokens=512, few_shot_prompt="", token_budget=None):
        effective_budget = token_budget if token_budget is not None else max_new_tokens

        if self.few_shot and few_shot_prompt:
            prompt = few_shot_prompt + f"\nQuestion: {question}\nLet's think step by step.\n"
        else:
            prompt = f"Question: {question}\nLet's think step by step.\n"

        text, logprobs, comp_tokens = _generate(
            model, tokenizer, prompt, max_new_tokens=effective_budget,
        )
        answer = _extract_answer(text)
        steps = _split_rationale_into_steps(text)
        confidence = math.exp(sum(logprobs) / max(len(logprobs), 1)) if logprobs else 0.0

        return StrategyResult(
            prediction=answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=len(tokenizer.encode(prompt)) + comp_tokens,
            prompt_tokens=len(tokenizer.encode(prompt)),
            completion_tokens=comp_tokens,
            logprobs=logprobs,
            step_metadata=[{"step": i, "content": s} for i, s in enumerate(steps)],
            strategy_name=self.name,
            reasoning_trace=text,
        )


class SelfConsistency(InferenceStrategy):
    name = "self_consistency"

    def __init__(self, k_samples: int = 5, temperature: float = 0.7):
        self.k_samples = k_samples
        self.temperature = temperature

    def run(self, model, tokenizer, question, max_new_tokens=512, few_shot_prompt="", token_budget=None):
        if few_shot_prompt:
            prompt = few_shot_prompt + f"\nQuestion: {question}\nLet's think step by step.\n"
        else:
            prompt = f"Question: {question}\nLet's think step by step.\n"

        if token_budget is not None:
            k_samples = max(1, token_budget // max(max_new_tokens, 1))
            per_sample_budget = max(32, token_budget // max(k_samples, 1))
        else:
            k_samples = self.k_samples
            per_sample_budget = max_new_tokens

        answers = []
        all_logprobs = []
        total_comp_tokens = 0
        traces = []

        for _ in range(k_samples):
            text, logprobs, comp_tokens = _generate(
                model, tokenizer, prompt,
                max_new_tokens=per_sample_budget,
                temperature=self.temperature,
                do_sample=True,
            )
            answer = _extract_answer(text)
            answers.append(answer)
            all_logprobs.append(logprobs)
            total_comp_tokens += comp_tokens
            traces.append(text)

        vote_counts = Counter(answers)
        majority_answer, majority_count = vote_counts.most_common(1)[0]
        agreement = majority_count / k_samples

        prompt_tokens = len(tokenizer.encode(prompt))
        return StrategyResult(
            prediction=majority_answer,
            confidence=agreement,
            total_tokens=prompt_tokens * k_samples + total_comp_tokens,
            prompt_tokens=prompt_tokens * k_samples,
            completion_tokens=total_comp_tokens,
            logprobs=all_logprobs[0] if all_logprobs else [],
            step_metadata=[{"sample": i, "answer": a, "trace": t} for i, (a, t) in enumerate(zip(answers, traces))],
            strategy_name=self.name,
            reasoning_trace=traces[0] if traces else "",
        )


class AnswerAnchoredBackwardCloze(InferenceStrategy):
    name = "backward_cloze"

    def __init__(self, n_candidates: int = 3):
        self.n_candidates = n_candidates

    def run(self, model, tokenizer, question, max_new_tokens=512, few_shot_prompt="", token_budget=None):
        n_candidates = self.n_candidates

        # Phase 1: Generate candidate answers via forward CoT with sampling
        cot_prompt = f"Question: {question}\nLet's think step by step.\n"
        if few_shot_prompt:
            cot_prompt = few_shot_prompt + f"\nQuestion: {question}\nLet's think step by step.\n"

        candidates = []
        for _ in range(n_candidates):
            text, lps, comp = _generate(
                model, tokenizer, cot_prompt,
                max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True,
            )
            answer = _extract_answer(text)
            avg_lp = sum(lps) / max(len(lps), 1) if lps else -10.0
            candidates.append((answer, avg_lp, text, lps, comp))

        # Phase 2: Backward verification — anchor on majority candidate, verify reasoning
        answer_counts = Counter(c[0] for c in candidates)
        majority_answer = answer_counts.most_common(1)[0][0] if answer_counts else candidates[0][0]

        verify_prompt = (
            f"Question: {question}\n\n"
            f"A student claims the answer is {majority_answer}. "
            f"Verify this by solving the problem step by step. "
            f"If the answer {majority_answer} is correct, confirm it. "
            f"If it is wrong, provide the correct answer.\n"
            f"End with: The answer is <your answer>.\n"
        )

        verify_text, verify_lps, verify_comp = _generate(
            model, tokenizer, verify_prompt, max_new_tokens=max_new_tokens,
        )

        verified_answer = _extract_answer(verify_text)
        final_answer = verified_answer if verified_answer else majority_answer

        all_lps = candidates[0][3] + verify_lps
        confidence = math.exp(sum(all_lps) / max(len(all_lps), 1)) if all_lps else 0.0
        total_comp = sum(c[4] for c in candidates) + verify_comp
        prompt_tokens = len(tokenizer.encode(cot_prompt)) * n_candidates + len(tokenizer.encode(verify_prompt))

        return StrategyResult(
            prediction=final_answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=prompt_tokens + total_comp,
            prompt_tokens=prompt_tokens,
            completion_tokens=total_comp,
            logprobs=all_lps,
            step_metadata=[
                {"candidates": [c[0] for c in candidates]},
                {"best_candidate": majority_answer},
                {"verified_answer": verified_answer},
            ],
            strategy_name=self.name,
            reasoning_trace=f"CANDIDATE GENERATION:\n{candidates[0][2]}\n\nVERIFICATION:\n{verify_text}",
        )


class UncertaintyTargetedMaskedRepair(InferenceStrategy):
    name = "uncertainty_masked_repair"

    def __init__(self, mask_fraction: float = 0.4, entropy_window: int = 5):
        self.mask_fraction = mask_fraction
        self.entropy_window = entropy_window

    def run(self, model, tokenizer, question, max_new_tokens=512, few_shot_prompt="", token_budget=None):
        cot_prompt = f"Question: {question}\nLet's think step by step.\n"
        if few_shot_prompt:
            cot_prompt = few_shot_prompt + f"\nQuestion: {question}\nLet's think step by step.\n"

        first_text, first_logprobs, first_comp = _generate(
            model, tokenizer, cot_prompt, max_new_tokens=max_new_tokens,
        )

        steps = _split_rationale_into_steps(first_text)
        if not steps or not first_logprobs:
            answer = _extract_answer(first_text)
            confidence = math.exp(sum(first_logprobs) / max(len(first_logprobs), 1)) if first_logprobs else 0.0
            return StrategyResult(
                prediction=answer,
                confidence=float(np.clip(confidence, 0, 1)),
                total_tokens=len(tokenizer.encode(cot_prompt)) + first_comp,
                prompt_tokens=len(tokenizer.encode(cot_prompt)),
                completion_tokens=first_comp,
                logprobs=first_logprobs,
                strategy_name=self.name,
                reasoning_trace=first_text,
            )

        entropies = _compute_token_entropy(first_logprobs, self.entropy_window)

        tokens_per_step = max(len(first_logprobs) // max(len(steps), 1), 1)
        step_uncertainties = []
        for i in range(len(steps)):
            start = i * tokens_per_step
            end = min((i + 1) * tokens_per_step, len(entropies))
            step_ent = np.mean(entropies[start:end]) if start < end else 0.0
            step_uncertainties.append((i, float(step_ent)))

        step_uncertainties.sort(key=lambda x: x[1], reverse=True)
        n_mask = max(1, int(len(steps) * self.mask_fraction))
        masked_indices = sorted([idx for idx, _ in step_uncertainties[:n_mask]])

        visible_steps = []
        for i, step in enumerate(steps):
            if i in masked_indices:
                visible_steps.append(f"Step {i + 1}: [UNCERTAIN - NEEDS REPAIR]")
            else:
                visible_steps.append(f"Step {i + 1}: {step}")

        repair_prompt = (
            f"Question: {question}\n\n"
            f"A previous attempt at solving this produced the following partial solution. "
            f"Steps marked [UNCERTAIN] may contain errors:\n\n"
            + "\n".join(visible_steps)
            + "\n\nRewrite the solution with the uncertain steps corrected. "
            f"End with: The answer is <your answer>.\n"
        )

        repair_text, repair_logprobs, repair_comp = _generate(
            model, tokenizer, repair_prompt, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True,
        )

        repaired_answer = _extract_answer(repair_text)
        if not repaired_answer:
            repaired_answer = _extract_answer(first_text)

        all_logprobs = first_logprobs + repair_logprobs
        confidence = math.exp(sum(all_logprobs) / max(len(all_logprobs), 1)) if all_logprobs else 0.0

        prompt_tokens = len(tokenizer.encode(cot_prompt)) + len(tokenizer.encode(repair_prompt))
        return StrategyResult(
            prediction=repaired_answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=prompt_tokens + first_comp + repair_comp,
            prompt_tokens=prompt_tokens,
            completion_tokens=first_comp + repair_comp,
            logprobs=all_logprobs,
            step_metadata=[
                {"masked_indices": masked_indices},
                {"step_uncertainties": step_uncertainties},
                {"original_steps": steps},
            ],
            strategy_name=self.name,
            reasoning_trace=f"FIRST PASS:\n{first_text}\n\nREPAIR:\n{repair_text}",
        )


class RandomSpanMaskedRepair(UncertaintyTargetedMaskedRepair):
    name = "random_masked_repair"

    def run(self, model, tokenizer, question, max_new_tokens=512, few_shot_prompt="", token_budget=None):
        cot_prompt = f"Question: {question}\nLet's think step by step.\n"
        if few_shot_prompt:
            cot_prompt = few_shot_prompt + f"\nQuestion: {question}\nLet's think step by step.\n"

        first_text, first_logprobs, first_comp = _generate(
            model, tokenizer, cot_prompt, max_new_tokens=max_new_tokens,
        )

        steps = _split_rationale_into_steps(first_text)
        if not steps:
            answer = _extract_answer(first_text)
            confidence = math.exp(sum(first_logprobs) / max(len(first_logprobs), 1)) if first_logprobs else 0.0
            return StrategyResult(
                prediction=answer,
                confidence=float(np.clip(confidence, 0, 1)),
                total_tokens=len(tokenizer.encode(cot_prompt)) + first_comp,
                prompt_tokens=len(tokenizer.encode(cot_prompt)),
                completion_tokens=first_comp,
                logprobs=first_logprobs,
                strategy_name=self.name,
                reasoning_trace=first_text,
            )

        rng = np.random.default_rng(hash(question) % (2**32))
        n_mask = max(1, int(len(steps) * self.mask_fraction))
        masked_indices = sorted(rng.choice(len(steps), size=min(n_mask, len(steps)), replace=False).tolist())

        visible_steps = []
        for i, step in enumerate(steps):
            if i in masked_indices:
                visible_steps.append(f"Step {i + 1}: [MASKED]")
            else:
                visible_steps.append(f"Step {i + 1}: {step}")

        repair_prompt = (
            f"Question: {question}\n\n"
            f"A previous attempt at solving this produced the following partial solution. "
            f"Some steps are [MASKED] and need to be filled in:\n\n"
            + "\n".join(visible_steps)
            + "\n\nFill in the missing steps and solve the problem completely. "
            f"End with: The answer is <your answer>.\n"
        )

        repair_text, repair_logprobs, repair_comp = _generate(
            model, tokenizer, repair_prompt, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True,
        )

        answer = _extract_answer(repair_text) or _extract_answer(first_text)
        all_logprobs = first_logprobs + repair_logprobs
        confidence = math.exp(sum(all_logprobs) / max(len(all_logprobs), 1)) if all_logprobs else 0.0

        prompt_tokens = len(tokenizer.encode(cot_prompt)) + len(tokenizer.encode(repair_prompt))
        return StrategyResult(
            prediction=answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=prompt_tokens + first_comp + repair_comp,
            prompt_tokens=prompt_tokens,
            completion_tokens=first_comp + repair_comp,
            logprobs=all_logprobs,
            step_metadata=[{"masked_indices": masked_indices, "random": True}],
            strategy_name=self.name,
            reasoning_trace=f"FIRST PASS:\n{first_text}\n\nREPAIR:\n{repair_text}",
        )


class FullRationaleRegeneration(InferenceStrategy):
    name = "full_regeneration"

    def run(self, model, tokenizer, question, max_new_tokens=512, few_shot_prompt="", token_budget=None):
        cot_prompt = f"Question: {question}\nLet's think step by step.\n"
        if few_shot_prompt:
            cot_prompt = few_shot_prompt + f"\nQuestion: {question}\nLet's think step by step.\n"

        first_text, first_lps, first_comp = _generate(
            model, tokenizer, cot_prompt, max_new_tokens=max_new_tokens,
        )

        first_answer = _extract_answer(first_text)
        critique_prompt = (
            f"Question: {question}\n\n"
            f"A student wrote this solution:\n{first_text}\n\n"
            f"Review this solution for errors and write a corrected, complete solution. "
            f"End with: The answer is <your answer>.\n"
        )

        second_text, second_lps, second_comp = _generate(
            model, tokenizer, critique_prompt, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True,
        )

        answer = _extract_answer(second_text) or first_answer
        all_lps = first_lps + second_lps
        confidence = math.exp(sum(all_lps) / max(len(all_lps), 1)) if all_lps else 0.0

        prompt_tokens = len(tokenizer.encode(cot_prompt)) + len(tokenizer.encode(critique_prompt))
        return StrategyResult(
            prediction=answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=prompt_tokens + first_comp + second_comp,
            prompt_tokens=prompt_tokens,
            completion_tokens=first_comp + second_comp,
            logprobs=all_lps,
            strategy_name=self.name,
            reasoning_trace=f"FIRST:\n{first_text}\n\nREVISED:\n{second_text}",
        )


class HierarchicalMaskedRepair(UncertaintyTargetedMaskedRepair):
    name = "hierarchical_repair"

    def __init__(self, mask_fraction: float = 0.3):
        super().__init__(mask_fraction=mask_fraction)

    def run(self, model, tokenizer, question, max_new_tokens=512, few_shot_prompt="", token_budget=None):
        cot_prompt = f"Question: {question}\nLet's think step by step.\n"
        if few_shot_prompt:
            cot_prompt = few_shot_prompt + f"\nQuestion: {question}\nLet's think step by step.\n"

        first_text, first_logprobs, first_comp = _generate(
            model, tokenizer, cot_prompt, max_new_tokens=max_new_tokens,
        )

        steps = _split_rationale_into_steps(first_text)
        if len(steps) < 2:
            answer = _extract_answer(first_text)
            confidence = math.exp(sum(first_logprobs) / max(len(first_logprobs), 1)) if first_logprobs else 0.0
            return StrategyResult(
                prediction=answer,
                confidence=float(np.clip(confidence, 0, 1)),
                total_tokens=len(tokenizer.encode(cot_prompt)) + first_comp,
                prompt_tokens=len(tokenizer.encode(cot_prompt)),
                completion_tokens=first_comp,
                logprobs=first_logprobs,
                strategy_name=self.name,
                reasoning_trace=first_text,
            )

        entropies = _compute_token_entropy(first_logprobs, window=5)
        tokens_per_step = max(len(first_logprobs) // max(len(steps), 1), 1)

        step_scores = []
        for i in range(len(steps)):
            start = i * tokens_per_step
            end = min((i + 1) * tokens_per_step, len(entropies))
            ent = np.mean(entropies[start:end]) if start < end else 0.0
            downstream_count = len(steps) - i - 1
            bottleneck_score = float(ent) * (1.0 + 0.5 * downstream_count / max(len(steps), 1))
            step_scores.append((i, bottleneck_score))

        step_scores.sort(key=lambda x: x[1], reverse=True)
        n_mask = max(1, int(len(steps) * self.mask_fraction))
        masked_indices = sorted([idx for idx, _ in step_scores[:n_mask]])

        visible_steps = []
        for i, step in enumerate(steps):
            if i in masked_indices:
                visible_steps.append(f"Step {i + 1}: [BOTTLENECK - NEEDS VERIFICATION]")
            else:
                visible_steps.append(f"Step {i + 1}: {step}")

        repair_prompt = (
            f"Question: {question}\n\n"
            f"A previous attempt at solving this flagged some critical steps as potentially incorrect:\n\n"
            + "\n".join(visible_steps)
            + "\n\nCarefully verify and correct the flagged steps, then solve the problem. "
            f"End with: The answer is <your answer>.\n"
        )

        repair_text, repair_lps, repair_comp = _generate(
            model, tokenizer, repair_prompt, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True,
        )

        answer = _extract_answer(repair_text) or _extract_answer(first_text)
        all_lps = first_logprobs + repair_lps
        confidence = math.exp(sum(all_lps) / max(len(all_lps), 1)) if all_lps else 0.0

        prompt_tokens = len(tokenizer.encode(cot_prompt)) + len(tokenizer.encode(repair_prompt))
        return StrategyResult(
            prediction=answer,
            confidence=float(np.clip(confidence, 0, 1)),
            total_tokens=prompt_tokens + first_comp + repair_comp,
            prompt_tokens=prompt_tokens,
            completion_tokens=first_comp + repair_comp,
            logprobs=all_lps,
            step_metadata=[{"bottleneck_indices": masked_indices, "scores": step_scores}],
            strategy_name=self.name,
            reasoning_trace=f"FIRST:\n{first_text}\n\nHIERARCHICAL REPAIR:\n{repair_text}",
        )


class CLOXAdaptive(InferenceStrategy):
    name = "clox_adaptive"

    def __init__(
        self,
        pilot_samples: int = 3,
        sc_samples: int = 5,
        mask_fraction: float = 0.25,
        temperature: float = 0.7,
    ):
        self.pilot_samples = pilot_samples
        self.sc_samples = sc_samples
        self.mask_fraction = mask_fraction
        self.temperature = temperature
        self.topology_estimator = TopologyEstimator(pilot_samples=pilot_samples)
        self.strategies: dict[str, InferenceStrategy] = {
            "standard_cot": StandardCoT(few_shot=True),
            "self_consistency": SelfConsistency(k_samples=sc_samples, temperature=temperature),
            "masked_repair": UncertaintyTargetedMaskedRepair(mask_fraction=mask_fraction),
            "hierarchical_repair": HierarchicalMaskedRepair(mask_fraction=mask_fraction),
        }

    def _run_pilot(self, model, tokenizer, question, max_new_tokens, few_shot_prompt):
        pilot_budget = max(max_new_tokens // 2, 128)  # Each pilot gets half budget (enough for topology)
        traces = []
        all_step_logprobs = []
        pilot_answers = []

        cot_prompt = f"Question: {question}\nLet's think step by step.\n"
        if few_shot_prompt:
            cot_prompt = few_shot_prompt + f"\nQuestion: {question}\nLet's think step by step.\n"

        prompt_tokens = len(tokenizer.encode(cot_prompt))

        for _ in range(self.pilot_samples):
            text, logprobs, _ = _generate(
                model, tokenizer, cot_prompt,
                max_new_tokens=pilot_budget,
                temperature=self.temperature,
                do_sample=True,
            )
            steps = _split_rationale_into_steps(text)
            reasoning_steps = []
            tokens_per_step = max(len(logprobs) // max(len(steps), 1), 1)
            for i, s in enumerate(steps):
                start = i * tokens_per_step
                end = min((i + 1) * tokens_per_step, len(logprobs))
                step_lps = logprobs[start:end]
                entropy = -sum(math.exp(lp) * lp for lp in step_lps) / max(len(step_lps), 1) if step_lps else 0.0
                reasoning_steps.append(ReasoningStep(
                    index=i, content=s, logprobs=step_lps, entropy=entropy,
                ))

            answer = _extract_answer(text)
            pilot_answers.append(answer)
            traces.append(ReasoningTrace(
                steps=reasoning_steps,
                final_answer=answer,
                total_tokens=prompt_tokens + len(logprobs),
                is_correct=False,
            ))
            all_step_logprobs.append(logprobs)

        majority_answer = Counter(pilot_answers).most_common(1)[0][0] if pilot_answers else ""
        for trace in traces:
            trace.is_correct = (trace.final_answer == majority_answer)

        return traces, all_step_logprobs

    def run(self, model, tokenizer, question, max_new_tokens=512, few_shot_prompt="", token_budget=None):
        effective_budget = token_budget if token_budget is not None else max_new_tokens

        pilot_traces, pilot_logprobs = self._run_pilot(
            model, tokenizer, question, effective_budget, few_shot_prompt,
        )

        topology = self.topology_estimator.estimate(pilot_traces, pilot_logprobs)
        selected_name = topology.strategy_recommendation
        if selected_name == "adaptive":
            selected_name = "masked_repair"

        selected_strategy = self.strategies.get(selected_name, self.strategies["standard_cot"])

        # Pilot is overhead — selected strategy gets full budget
        result = selected_strategy.run(
            model, tokenizer, question,
            max_new_tokens=max_new_tokens,
            few_shot_prompt=few_shot_prompt,
        )

        result.strategy_name = f"clox_adaptive({selected_name})"
        result.pilot_traces = pilot_traces
        result.topology = topology
        result.total_tokens += sum(t.total_tokens for t in pilot_traces)
        result.step_metadata.append({
            "topology": {
                "epl": topology.epl,
                "recoverability": topology.local_recoverability,
                "n_steps": topology.n_steps,
                "selected_strategy": selected_name,
            }
        })

        return result


STRATEGY_REGISTRY: dict[str, type[InferenceStrategy]] = {
    "standard_cot": StandardCoT,
    "self_consistency": SelfConsistency,
    "backward_cloze": AnswerAnchoredBackwardCloze,
    "uncertainty_masked_repair": UncertaintyTargetedMaskedRepair,
    "random_masked_repair": RandomSpanMaskedRepair,
    "full_regeneration": FullRationaleRegeneration,
    "hierarchical_repair": HierarchicalMaskedRepair,
    "clox_adaptive": CLOXAdaptive,
}


def build_strategy(name: str, **kwargs) -> InferenceStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    return cls(**kwargs)
