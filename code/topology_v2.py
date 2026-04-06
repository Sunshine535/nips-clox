"""Topology estimation for reasoning traces.

Computes two key structural properties:
  r̄ (local recoverability): Can individual steps be reconstructed from context?
  ℓ (error propagation length): How far do errors cascade?

These determine which inference strategy is optimal (Theorems 1-3).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from engine import GenerationOutput, VLLMEngine, extract_answer, split_into_steps


@dataclass
class TopologyProfile:
    r_bar: float              # local recoverability ∈ [0, 1]
    epl: float                # error propagation length
    n_steps: float            # mean number of reasoning steps
    step_entropies: list[float]  # per-step entropy values
    step_agreements: list[float]  # per-step agreement across samples
    strategy: str             # recommended strategy


@dataclass
class TraceAnalysis:
    """Analyzed reasoning trace with step-level metrics."""
    text: str
    steps: list[str]
    answer: str
    step_entropies: list[float]   # mean token entropy per step
    step_logprobs: list[float]    # mean token logprob per step
    token_entropies: list[float]  # raw token-level entropies
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


def analyze_trace(output: GenerationOutput) -> TraceAnalysis:
    """Extract step-level metrics from a generation output."""
    steps = split_into_steps(output.text)
    answer = extract_answer(output.text)
    token_ents = output.token_entropy()

    if not steps:
        return TraceAnalysis(
            text=output.text, steps=[], answer=answer,
            step_entropies=[], step_logprobs=[], token_entropies=token_ents,
            total_tokens=output.total_tokens,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
        )

    n_tokens = len(output.logprobs)
    tokens_per_step = max(n_tokens // len(steps), 1)

    step_ents = []
    step_lps = []
    for i in range(len(steps)):
        start = i * tokens_per_step
        end = min((i + 1) * tokens_per_step, n_tokens)
        if start < end:
            step_ents.append(float(np.mean(token_ents[start:end])))
            step_lps.append(float(np.mean(output.logprobs[start:end])))
        else:
            step_ents.append(0.0)
            step_lps.append(0.0)

    return TraceAnalysis(
        text=output.text, steps=steps, answer=answer,
        step_entropies=step_ents, step_logprobs=step_lps,
        token_entropies=token_ents,
        total_tokens=output.total_tokens,
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
    )


def compute_step_agreement(traces: list[TraceAnalysis], step_idx: int) -> float:
    """Jaccard agreement of step content across traces at a given position."""
    contents = [t.steps[step_idx] for t in traces if step_idx < len(t.steps)]
    if len(contents) < 2:
        return 1.0
    agreements = 0
    comparisons = 0
    for i in range(len(contents)):
        for j in range(i + 1, len(contents)):
            words_i = set(contents[i].lower().split())
            words_j = set(contents[j].lower().split())
            union = words_i | words_j
            if union:
                agreements += len(words_i & words_j) / len(union)
            comparisons += 1
    return agreements / max(comparisons, 1)


def estimate_recoverability(
    traces: list[TraceAnalysis],
    engine: VLLMEngine | None = None,
    question: str = "",
    few_shot: str = "",
) -> tuple[float, list[float]]:
    """Estimate local recoverability r̄.

    Method 1 (fast): Cross-sample step agreement weighted by confidence.
    Method 2 (accurate, requires engine): Mask-and-regenerate each step,
    measure if the model can recover it from surrounding context.

    Returns (r_bar, per_step_recoverability).
    """
    if not traces or not traces[0].steps:
        return 0.5, []

    min_steps = min(len(t.steps) for t in traces)
    if min_steps == 0:
        return 0.5, []

    # Method 1: Agreement-based estimation
    per_step_agreement = []
    for idx in range(min_steps):
        agreement = compute_step_agreement(traces, idx)
        per_step_agreement.append(agreement)

    # Weight by confidence (higher logprob = more reliable agreement signal)
    mean_confidence = []
    for idx in range(min_steps):
        step_confs = [math.exp(t.step_logprobs[idx]) for t in traces
                      if idx < len(t.step_logprobs)]
        mean_confidence.append(float(np.mean(step_confs)) if step_confs else 0.5)

    # Combine: r_i = 0.6 * agreement + 0.4 * confidence
    per_step_r = []
    for ag, conf in zip(per_step_agreement, mean_confidence):
        r_i = 0.6 * ag + 0.4 * conf
        per_step_r.append(float(np.clip(r_i, 0.0, 1.0)))

    # Method 2: Mask-and-regenerate (if engine available)
    if engine is not None and question and min_steps >= 3:
        regeneration_scores = _mask_regenerate_test(
            engine, traces[0], question, few_shot,
        )
        if regeneration_scores:
            # Blend: 50% agreement-based + 50% regeneration-based
            for i in range(min(len(per_step_r), len(regeneration_scores))):
                per_step_r[i] = 0.5 * per_step_r[i] + 0.5 * regeneration_scores[i]

    r_bar = float(np.mean(per_step_r)) if per_step_r else 0.5
    return r_bar, per_step_r


def _mask_regenerate_test(
    engine: VLLMEngine,
    trace: TraceAnalysis,
    question: str,
    few_shot: str = "",
    n_regenerations: int = 3,
) -> list[float]:
    """Mask each step and test if the model can regenerate it."""
    steps = trace.steps
    if len(steps) < 3:
        return []

    # Only test a subset of steps to save compute
    test_indices = list(range(1, len(steps) - 1))  # skip first and last
    if len(test_indices) > 5:
        rng = np.random.default_rng(42)
        test_indices = sorted(rng.choice(test_indices, size=5, replace=False).tolist())

    prompts = []
    for idx in test_indices:
        visible = []
        for i, step in enumerate(steps):
            if i == idx:
                visible.append(f"Step {i+1}: [MISSING - fill this in]")
            else:
                visible.append(f"Step {i+1}: {step}")

        prompt = (
            f"{few_shot}\nQuestion: {question}\n\n"
            f"Below is a partial solution with one step missing. "
            f"Fill in the missing step:\n\n"
            + "\n".join(visible)
            + "\n\nFill in the missing step and provide the complete content for that step only:\n"
        )
        prompts.append(prompt)

    if not prompts:
        return []

    outputs = engine.generate_batch(
        prompts * n_regenerations if n_regenerations > 1 else prompts,
        max_tokens=256, temperature=0.3 if n_regenerations > 1 else 0.0,
    )

    scores = []
    for i, idx in enumerate(test_indices):
        original_words = set(steps[idx].lower().split())
        regen_agreements = []
        for r in range(n_regenerations):
            out_idx = i + r * len(test_indices) if n_regenerations > 1 else i
            if out_idx < len(outputs):
                regen_words = set(outputs[out_idx].text.lower().split())
                union = original_words | regen_words
                if union:
                    regen_agreements.append(len(original_words & regen_words) / len(union))
        scores.append(float(np.mean(regen_agreements)) if regen_agreements else 0.5)

    # Map test_indices scores back to full step list
    full_scores = [0.5] * len(steps)
    for idx, score in zip(test_indices, scores):
        full_scores[idx] = score
    # Interpolate first/last from neighbors
    if scores:
        full_scores[0] = scores[0]
        full_scores[-1] = scores[-1]

    return full_scores


def estimate_epl(traces: list[TraceAnalysis]) -> float:
    """Estimate error propagation length ℓ.

    Looks for correlated disagreements across steps: if step i disagrees
    across traces AND step j also disagrees, and this happens consistently,
    the distance |j - i| is an EPL sample.
    """
    if not traces or len(traces) < 2:
        return 3.0

    min_steps = min(len(t.steps) for t in traces)
    if min_steps <= 1:
        return 1.0

    # Per-step agreement
    agreements = []
    for idx in range(min_steps):
        agreements.append(compute_step_agreement(traces, idx))

    # Find disagreement pairs (both steps have agreement < 0.85)
    disagreement_threshold = 0.85
    disagreeing_steps = [i for i, a in enumerate(agreements) if a < disagreement_threshold]

    if len(disagreeing_steps) < 2:
        # No correlated errors → short EPL
        return 1.0

    # EPL = mean distance between consecutive disagreeing steps
    # This captures how far errors propagate
    distances = []
    for i in range(len(disagreeing_steps)):
        for j in range(i + 1, len(disagreeing_steps)):
            distances.append(abs(disagreeing_steps[j] - disagreeing_steps[i]))

    # Also measure: do answer disagreements correlate with early step disagreements?
    answers = [t.answer for t in traces]
    answer_diversity = len(set(answers)) / max(len(answers), 1)

    # High answer diversity + long error distances = high EPL
    raw_epl = float(np.mean(distances)) if distances else 1.0

    # Scale by answer diversity (more diverse answers → errors propagate to the end)
    epl = raw_epl * (0.5 + 0.5 * answer_diversity)

    return float(np.clip(epl, 1.0, min_steps))


def estimate_topology(
    engine: VLLMEngine,
    question: str,
    few_shot: str = "",
    n_pilot: int = 8,
    max_tokens: int = 512,
    temperature: float = 0.7,
    do_regeneration_test: bool = True,
) -> TopologyProfile:
    """Full topology estimation for a question.

    Generates n_pilot traces, analyzes step-level structure,
    computes (r̄, ℓ), and recommends a strategy.
    """
    prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step.\n"

    outputs = engine.generate_multi(
        prompt, n=n_pilot, max_tokens=max_tokens, temperature=temperature,
    )
    traces = [analyze_trace(o) for o in outputs]

    r_bar, per_step_r = estimate_recoverability(
        traces,
        engine=engine if do_regeneration_test else None,
        question=question,
        few_shot=few_shot,
    )
    epl = estimate_epl(traces)

    step_counts = [len(t.steps) for t in traces]
    n_steps = float(np.mean(step_counts)) if step_counts else 5.0

    all_step_ents = []
    for t in traces:
        all_step_ents.extend(t.step_entropies)

    step_agreements = []
    min_steps = min(len(t.steps) for t in traces) if traces else 0
    for idx in range(min_steps):
        step_agreements.append(compute_step_agreement(traces, idx))

    strategy = recommend_strategy(r_bar, epl, n_steps)

    return TopologyProfile(
        r_bar=r_bar,
        epl=epl,
        n_steps=n_steps,
        step_entropies=all_step_ents,
        step_agreements=step_agreements,
        strategy=strategy,
    )


def recommend_strategy(r_bar: float, epl: float, n_steps: float) -> str:
    """Map topology to optimal strategy based on Theorems 1-3.

    Theorem 1: ℓ ≤ O(log n), r̄ ≥ 1-δ → masked repair
    Theorem 2: ℓ ≥ Ω(n), r̄ ≤ 1/2 → self-consistency
    Theorem 3: No fixed strategy is universally optimal
    """
    norm_epl = epl / max(n_steps, 1.0)
    log_threshold = math.log(max(n_steps, 2)) / max(n_steps, 1)

    # Regime 1: Short EPL + high recoverability → masking advantage
    if norm_epl <= max(log_threshold, 0.3) and r_bar >= 0.65:
        return "targeted_repair"

    # Regime 2: Long EPL + low recoverability → resampling advantage
    if norm_epl >= 0.5 and r_bar <= 0.45:
        return "self_consistency"

    # Regime 3: Long EPL + high recoverability → hierarchical repair
    if norm_epl >= 0.4 and r_bar >= 0.6:
        return "hierarchical_repair"

    # Regime 4: Short EPL + low recoverability → standard CoT sufficient
    if norm_epl <= 0.3 and r_bar <= 0.4:
        return "standard_cot"

    # Boundary region → adaptive (use both and pick better)
    return "adaptive"


def batch_estimate_topology(
    engine: VLLMEngine,
    questions: list[str],
    few_shot: str = "",
    n_pilot: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[TopologyProfile]:
    """Efficient batch topology estimation.

    Generates all pilot traces in one batch for throughput.
    """
    prompts = [
        f"{few_shot}\nQuestion: {q}\nLet's think step by step.\n"
        for q in questions
    ]

    # Generate n_pilot samples per prompt in one batch
    all_outputs = engine.generate(
        prompts, max_tokens=max_tokens, temperature=temperature,
        n=n_pilot, logprobs=20,
    )

    profiles = []
    for q_idx, q_outputs in enumerate(all_outputs):
        traces = [analyze_trace(o) for o in q_outputs]
        r_bar, _ = estimate_recoverability(traces)
        epl = estimate_epl(traces)
        step_counts = [len(t.steps) for t in traces]
        n_steps = float(np.mean(step_counts)) if step_counts else 5.0

        strategy = recommend_strategy(r_bar, epl, n_steps)
        profiles.append(TopologyProfile(
            r_bar=r_bar, epl=epl, n_steps=n_steps,
            step_entropies=[], step_agreements=[],
            strategy=strategy,
        ))

    return profiles
