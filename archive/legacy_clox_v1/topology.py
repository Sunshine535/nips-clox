from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReasoningStep:
    index: int
    content: str
    logprobs: list[float] = field(default_factory=list)
    entropy: float = 0.0
    correct: bool | None = None
    dependencies: list[int] = field(default_factory=list)


@dataclass
class ReasoningTrace:
    steps: list[ReasoningStep]
    final_answer: str
    total_tokens: int
    is_correct: bool


@dataclass
class TopologyProfile:
    epl: float
    local_recoverability: float
    n_steps: float
    step_entropies: list[float]
    dependency_depth: float
    strategy_recommendation: str


def _split_rationale_into_steps(text: str) -> list[str]:
    delimiters = ["\n\n", "\nStep ", "\n- ", "\n* ", "\nTherefore", "\nSo ", "\nThus "]
    segments = [text]
    for delimiter in delimiters:
        new_segments = []
        for segment in segments:
            parts = segment.split(delimiter)
            for i, part in enumerate(parts):
                cleaned = part.strip()
                if cleaned:
                    new_segments.append(cleaned if i == 0 else delimiter.strip() + " " + cleaned)
        segments = new_segments
    return [s for s in segments if len(s.split()) >= 3]


def _compute_step_agreement(samples: list[list[str]], step_idx: int) -> float:
    if not samples or step_idx >= min(len(s) for s in samples):
        return 0.0
    step_contents = [s[step_idx] for s in samples if step_idx < len(s)]
    if len(step_contents) < 2:
        return 1.0
    agreements = 0
    comparisons = 0
    for i in range(len(step_contents)):
        for j in range(i + 1, len(step_contents)):
            words_i = set(step_contents[i].lower().split())
            words_j = set(step_contents[j].lower().split())
            union = words_i | words_j
            if union:
                agreements += len(words_i & words_j) / len(union)
            comparisons += 1
    return agreements / max(comparisons, 1)


def estimate_local_recoverability(
    pilot_traces: list[ReasoningTrace],
    step_logprobs: list[list[float]] | None = None,
) -> float:
    if len(pilot_traces) < 2:
        return 0.5

    step_lists = []
    for trace in pilot_traces:
        steps = [s.content for s in trace.steps] if trace.steps else _split_rationale_into_steps(trace.final_answer)
        step_lists.append(steps)

    min_steps = min(len(s) for s in step_lists)
    if min_steps == 0:
        return 0.5

    agreements = []
    for idx in range(min_steps):
        agreement = _compute_step_agreement(step_lists, idx)
        agreements.append(agreement)

    base_recoverability = float(np.mean(agreements)) if agreements else 0.5

    if step_logprobs:
        flat_probs = []
        for trace_probs in step_logprobs:
            for p in trace_probs:
                if np.isfinite(p):
                    flat_probs.append(math.exp(p))
        if flat_probs:
            confidence_factor = float(np.clip(np.mean(flat_probs), 0.0, 1.0))
            base_recoverability = 0.7 * base_recoverability + 0.3 * confidence_factor

    return float(np.clip(base_recoverability, 0.0, 1.0))


def estimate_epl(
    pilot_traces: list[ReasoningTrace],
    ground_truth: str | None = None,
) -> float:
    if len(pilot_traces) < 2:
        return 3.0

    step_lists = []
    for trace in pilot_traces:
        steps = [s.content for s in trace.steps] if trace.steps else _split_rationale_into_steps(trace.final_answer)
        step_lists.append(steps)

    min_steps = min(len(s) for s in step_lists)
    if min_steps <= 1:
        return 1.0

    error_correlations = []
    for i in range(min_steps):
        for j in range(i + 1, min_steps):
            agreement_i = _compute_step_agreement(step_lists, i)
            agreement_j = _compute_step_agreement(step_lists, j)
            disagreement_i = 1.0 - agreement_i
            disagreement_j = 1.0 - agreement_j
            if disagreement_i > 0.1 and disagreement_j > 0.1:
                error_correlations.append(abs(j - i))

    if error_correlations:
        raw_epl = float(np.mean(error_correlations))
    else:
        correct_count = sum(1 for t in pilot_traces if t.is_correct)
        error_rate = 1.0 - correct_count / max(len(pilot_traces), 1)
        raw_epl = min_steps * error_rate + 1.0

    return float(np.clip(raw_epl, 1.0, min_steps))


def recommend_strategy(epl: float, recoverability: float, n_steps: float) -> str:
    normalized_epl = epl / max(n_steps, 1.0)

    if recoverability >= 0.7 and normalized_epl <= 0.3:
        return "masked_repair"
    elif recoverability <= 0.4 and normalized_epl >= 0.5:
        return "self_consistency"
    elif recoverability >= 0.6 and normalized_epl >= 0.4:
        return "hierarchical_repair"
    elif recoverability <= 0.4 and normalized_epl <= 0.3:
        return "standard_cot"
    else:
        return "adaptive"


class TopologyEstimator:
    def __init__(self, pilot_samples: int = 3):
        self.pilot_samples = pilot_samples

    def estimate(
        self,
        pilot_traces: list[ReasoningTrace],
        step_logprobs: list[list[float]] | None = None,
        ground_truth: str | None = None,
    ) -> TopologyProfile:
        r_bar = estimate_local_recoverability(pilot_traces, step_logprobs)
        epl = estimate_epl(pilot_traces, ground_truth)

        step_counts = []
        all_entropies = []
        for trace in pilot_traces:
            if trace.steps:
                step_counts.append(len(trace.steps))
                all_entropies.extend([s.entropy for s in trace.steps])
            else:
                steps = _split_rationale_into_steps(trace.final_answer)
                step_counts.append(len(steps))

        n_steps = float(np.mean(step_counts)) if step_counts else 5.0
        step_entropies = all_entropies if all_entropies else [0.0]

        max_depth = max(
            (max((d for s in trace.steps for d in s.dependencies), default=0) if trace.steps else 0)
            for trace in pilot_traces
        ) if pilot_traces else 1.0

        strategy = recommend_strategy(epl, r_bar, n_steps)

        return TopologyProfile(
            epl=epl,
            local_recoverability=r_bar,
            n_steps=n_steps,
            step_entropies=step_entropies,
            dependency_depth=float(max_depth),
            strategy_recommendation=strategy,
        )


def compute_theoretical_error_bound(
    n_steps: int,
    per_step_error: float,
    strategy: str,
    recoverability: float = 0.5,
    epl: float = 3.0,
    masking_fraction: float = 0.25,
    sc_samples: int = 5,
) -> float:
    eps = max(per_step_error, 1e-10)

    if strategy == "standard_cot":
        return 1.0 - (1.0 - eps) ** n_steps

    if strategy == "self_consistency":
        p_correct_single = (1.0 - eps) ** n_steps
        p_majority_wrong = 0.0
        for k in range(sc_samples // 2 + 1):
            binom_coeff = math.comb(sc_samples, k)
            p_majority_wrong += binom_coeff * (p_correct_single ** k) * ((1 - p_correct_single) ** (sc_samples - k))
        return float(np.clip(p_majority_wrong, 0.0, 1.0))

    if strategy in ("masked_repair", "uncertainty_targeted"):
        alpha = min(masking_fraction, epl / max(n_steps, 1))
        e_mask = n_steps * eps * (1.0 - alpha * recoverability) + alpha * n_steps * eps * eps
        return float(np.clip(e_mask, 0.0, 1.0))

    if strategy == "hierarchical_repair":
        bottleneck_fraction = min(0.5, epl / max(n_steps, 1))
        e_hier = n_steps * eps * (1.0 - bottleneck_fraction * recoverability * 0.8)
        return float(np.clip(e_hier, 0.0, 1.0))

    return 1.0 - (1.0 - eps) ** n_steps
