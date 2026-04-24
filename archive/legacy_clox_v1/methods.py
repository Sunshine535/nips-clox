from __future__ import annotations

import math

import numpy as np

from data import answer_from_digits, base_reasoning_tokens
from utils import logsumexp, stable_int_seed

EPS = 1e-12

def safe_log(value: float) -> float:
    return math.log(max(float(value), EPS))

def compute_entropies(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPS, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)

def argmax_digits(probabilities: np.ndarray) -> list[int]:
    return [int(np.argmax(probability)) for probability in probabilities]

def assignment_logprob(probabilities: np.ndarray, digits) -> float:
    return float(sum(safe_log(probabilities[index][digit]) for index, digit in enumerate(digits)))

def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), EPS)
    logits = np.log(np.clip(probability, EPS, 1.0)) / temperature
    logits = logits - logits.max()
    weights = np.exp(logits)
    return weights / max(weights.sum(), EPS)

def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    radius = window // 2
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(float(np.mean(values[start:end])))
    return np.asarray(smoothed, dtype=float)

def enumerate_assignments(
    probabilities: np.ndarray,
    topk: int,
    beam_size: int,
    fixed_digits: dict[int, int] | None = None,
):
    fixed_digits = {} if fixed_digits is None else dict(fixed_digits)
    beam = [([], 0.0)]
    for position, probability in enumerate(probabilities):
        if position in fixed_digits:
            candidates = [int(fixed_digits[position])]
        else:
            candidates = [int(index) for index in np.argsort(probability)[-int(topk) :][::-1]]
        next_beam = []
        for prefix, score in beam:
            for digit in candidates:
                next_beam.append((prefix + [digit], score + safe_log(probability[digit])))
        next_beam.sort(key=lambda item: item[1], reverse=True)
        beam = next_beam[: int(beam_size)]
    return [(tuple(digits), float(score)) for digits, score in beam]

def aggregate_answer_statistics(
    example,
    probabilities: np.ndarray,
    hyperparameters: dict,
    fixed_digits: dict[int, int] | None = None,
):
    assignments = enumerate_assignments(
        probabilities=probabilities,
        topk=int(hyperparameters["beam_topk"]),
        beam_size=int(hyperparameters["beam_size"]),
        fixed_digits=fixed_digits,
    )
    answer_log_masses: dict[int, float] = {}
    best_assignment_by_answer: dict[int, tuple[float, tuple[int, ...]]] = {}
    for digits, score in assignments:
        answer = answer_from_digits(example.task_type, digits, hyperparameters)
        answer_log_masses[answer] = np.logaddexp(answer_log_masses.get(answer, -1e30), score)
        if answer not in best_assignment_by_answer or score > best_assignment_by_answer[answer][0]:
            best_assignment_by_answer[answer] = (score, digits)
    return answer_log_masses, best_assignment_by_answer, assignments

def answer_posterior(answer_log_masses: dict[int, float], answer: int) -> float:
    if not answer_log_masses or answer not in answer_log_masses:
        return 0.0
    denominator = logsumexp(answer_log_masses.values())
    if not np.isfinite(denominator):
        return 0.0
    return float(math.exp(answer_log_masses[answer] - denominator))

def perturb_positions(example, probabilities: np.ndarray, digits, positions, hyperparameters: dict) -> float:
    positions = [int(position) for position in positions]
    if not positions:
        return 0.0
    perturbed_digits = list(int(digit) for digit in digits)
    for position in positions:
        ranked = [int(index) for index in np.argsort(probabilities[position])[::-1]]
        for candidate_digit in ranked:
            if candidate_digit != perturbed_digits[position]:
                perturbed_digits[position] = candidate_digit
                break
    original_answer = answer_from_digits(example.task_type, digits, hyperparameters)
    perturbed_answer = answer_from_digits(example.task_type, perturbed_digits, hyperparameters)
    return float(original_answer != perturbed_answer)

def top2_margin(probability: np.ndarray) -> float:
    ranked = np.sort(np.asarray(probability, dtype=float))
    if ranked.size < 2:
        return 0.0
    return float(ranked[-1] - ranked[-2])

def answer_flip_risk(example, probabilities: np.ndarray, digits, position: int, hyperparameters: dict) -> float:
    ranked = [int(index) for index in np.argsort(probabilities[position])[::-1]]
    original_answer = answer_from_digits(example.task_type, digits, hyperparameters)
    original_digit = int(digits[position])
    risk = 0.0
    mass = 0.0
    for candidate_digit in ranked[:3]:
        if candidate_digit == original_digit:
            continue
        trial = list(int(value) for value in digits)
        trial[position] = candidate_digit
        changed = float(answer_from_digits(example.task_type, trial, hyperparameters) != original_answer)
        weight = float(probabilities[position][candidate_digit])
        risk += changed * weight
        mass += weight
    if mass <= EPS:
        return 0.0
    return float(risk / mass)

def select_best_answer_constrained_assignment(
    example,
    probabilities: np.ndarray,
    hyperparameters: dict,
    selected_positions: list[int],
    fixed_digits: dict[int, int],
    unrestricted_answer_log_masses: dict[int, float],
) -> tuple[tuple[int, ...], int, float, float]:
    candidate_answers = [
        int(answer)
        for answer, _ in sorted(
            unrestricted_answer_log_masses.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, int(hyperparameters["answer_candidates"]))]
    ]
    if not candidate_answers:
        digits = tuple(argmax_digits(probabilities))
        answer = answer_from_digits(example.task_type, digits, hyperparameters)
        return digits, int(answer), 0.0, 0.0

    best_tuple = None
    selected_positions = sorted(int(position) for position in selected_positions)

    for candidate_answer in candidate_answers:
        local_answer_log_masses, best_assignments, _ = aggregate_answer_statistics(
            example=example,
            probabilities=probabilities,
            hyperparameters=hyperparameters,
            fixed_digits=fixed_digits,
        )
        if candidate_answer not in best_assignments:
            continue

        digits = best_assignments[candidate_answer][1]
        local_posterior = answer_posterior(local_answer_log_masses, int(candidate_answer))
        global_posterior = answer_posterior(unrestricted_answer_log_masses, int(candidate_answer))
        normalized_logprob = assignment_logprob(probabilities, digits) / (
            len(probabilities) ** float(hyperparameters["score_normalization_alpha"])
        )

        fill_confidences = [float(probabilities[position][digits[position]]) for position in selected_positions]
        fill_consistency = float(np.mean(fill_confidences)) if fill_confidences else 1.0

        score = (
            normalized_logprob
            + 1.15 * safe_log(max(global_posterior, EPS))
            + 0.60 * safe_log(max(local_posterior, EPS))
            + 0.25 * safe_log(max(fill_consistency, EPS))
        )

        candidate = (
            score,
            digits,
            int(candidate_answer),
            float(fill_consistency),
            float(global_posterior),
        )
        if best_tuple is None or candidate[0] > best_tuple[0]:
            best_tuple = candidate

    if best_tuple is None:
        digits = tuple(argmax_digits(probabilities))
        answer = answer_from_digits(example.task_type, digits, hyperparameters)
        posterior = answer_posterior(unrestricted_answer_log_masses, int(answer))
        return digits, int(answer), 1.0, posterior

    return best_tuple[1], best_tuple[2], best_tuple[3], best_tuple[4]

def build_prediction_record(
    example,
    probabilities: np.ndarray,
    hyperparameters: dict,
    budget_regime: str,
    digits,
    selected_positions,
    repair_ratio: float,
    token_count: int,
    blank_fill_consistency: float,
    answer_posterior_value: float,
) -> dict:
    digits = tuple(int(digit) for digit in digits)
    selected_positions = sorted(int(position) for position in selected_positions)
    entropies = compute_entropies(probabilities)
    ranked_positions = [int(index) for index in np.argsort(entropies)[::-1]]
    chosen_probabilities = [float(probabilities[index][digit]) for index, digit in enumerate(digits)]
    chosen_logprobs = [safe_log(probability) for probability in chosen_probabilities]
    prediction = answer_from_digits(example.task_type, digits, hyperparameters)

    standard_sensitivity_by_k = {}
    for k in (1, 2):
        matched_positions = ranked_positions[: min(k, len(ranked_positions))]
        standard_sensitivity_by_k[k] = perturb_positions(
            example=example,
            probabilities=probabilities,
            digits=digits,
            positions=matched_positions,
            hyperparameters=hyperparameters,
        )

    return {
        "prediction": int(prediction),
        "correct": int(prediction == example.answer),
        "assignment": list(digits),
        "position_probabilities": chosen_probabilities,
        "position_logprobs": chosen_logprobs,
        "position_entropies": entropies.tolist(),
        "confidence": float(np.mean(chosen_probabilities)),
        "answer_posterior": float(np.clip(answer_posterior_value, 0.0, 1.0)),
        "tokens": int(token_count),
        "selected_positions": selected_positions,
        "repair_ratio": float(repair_ratio),
        "blank_fill_consistency": float(np.clip(blank_fill_consistency, 0.0, 1.0)),
        "sensitivity": float(
            perturb_positions(
                example=example,
                probabilities=probabilities,
                digits=digits,
                positions=selected_positions,
                hyperparameters=hyperparameters,
            )
        ),
        "standard_sensitivity_by_k": standard_sensitivity_by_k,
        "budget_regime": budget_regime,
    }

class BaseCondition:
    name = "BaseCondition"
    budget_regime = "tight_budget"

    def __init__(self, hyperparameters: dict, seed: int = 0) -> None:
        self.hyperparameters = hyperparameters
        self.seed = int(seed)

    def predict(self, example, probabilities: np.ndarray) -> dict:
        raise NotImplementedError

class StandardLeftToRightChainOfThought(BaseCondition):
    name = "StandardLeftToRightChainOfThought"
    budget_regime = "tight_budget"

    def predict(self, example, probabilities: np.ndarray) -> dict:
        digits = argmax_digits(probabilities)
        answer_log_masses, _, _ = aggregate_answer_statistics(
            example=example,
            probabilities=probabilities,
            hyperparameters=self.hyperparameters,
            fixed_digits=None,
        )
        prediction = answer_from_digits(example.task_type, digits, self.hyperparameters)
        return build_prediction_record(
            example=example,
            probabilities=probabilities,
            hyperparameters=self.hyperparameters,
            budget_regime=self.budget_regime,
            digits=digits,
            selected_positions=[],
            repair_ratio=0.0,
            token_count=base_reasoning_tokens(example.task_type, self.hyperparameters),
            blank_fill_consistency=1.0,
            answer_posterior_value=answer_posterior(answer_log_masses, int(prediction)),
        )

class BudgetMatchedSelfConsistencyConsensus(BaseCondition):
    name = "BudgetMatchedSelfConsistencyConsensus"
    budget_regime = "moderate_budget"

    def predict(self, example, probabilities: np.ndarray) -> dict:
        rng = np.random.default_rng(stable_int_seed(self.name, self.seed, example.example_id))
        sampled_digits = []
        sampled_answers = []
        sampled_scores = []

        for _ in range(int(self.hyperparameters["self_consistency_samples"])):
            digits = []
            for probability in probabilities:
                distribution = apply_temperature(
                    probability=probability,
                    temperature=float(self.hyperparameters["temperature_self_consistency"]),
                )
                digits.append(int(rng.choice(np.arange(len(distribution)), p=distribution)))
            sampled_digits.append(tuple(digits))
            sampled_answers.append(answer_from_digits(example.task_type, digits, self.hyperparameters))
            sampled_scores.append(assignment_logprob(probabilities, digits))

        best_vote = None
        for answer in sorted(set(sampled_answers)):
            vote_count = sampled_answers.count(answer)
            score_sum = sum(
                score for candidate_answer, score in zip(sampled_answers, sampled_scores) if candidate_answer == answer
            )
            candidate = (vote_count, score_sum, answer)
            if best_vote is None or candidate > best_vote:
                best_vote = candidate

        target_answer = int(best_vote[2])
        winning_indices = [index for index, answer in enumerate(sampled_answers) if answer == target_answer]
        best_index = max(winning_indices, key=lambda index: sampled_scores[index])
        agreement = float(sampled_answers.count(target_answer) / max(len(sampled_answers), 1))

        return build_prediction_record(
            example=example,
            probabilities=probabilities,
            hyperparameters=self.hyperparameters,
            budget_regime=self.budget_regime,
            digits=sampled_digits[best_index],
            selected_positions=[],
            repair_ratio=0.0,
            token_count=base_reasoning_tokens(example.task_type, self.hyperparameters)
            * int(self.hyperparameters["self_consistency_samples"]),
            blank_fill_consistency=agreement,
            answer_posterior_value=agreement,
        )

class AnswerAnchoredBackwardClozeReconstruction(BaseCondition):
    name = "AnswerAnchoredBackwardClozeReconstruction"
    budget_regime = "tight_budget"

    def __init__(self, hyperparameters: dict, seed: int = 0, fill_order: str = "backward") -> None:
        super().__init__(hyperparameters=hyperparameters, seed=seed)
        self.fill_order = str(fill_order)

    def predict(self, example, probabilities: np.ndarray) -> dict:
        unrestricted_answer_log_masses, _, _ = aggregate_answer_statistics(
            example=example,
            probabilities=probabilities,
            hyperparameters=self.hyperparameters,
            fixed_digits=None,
        )
        first_pass_digits = argmax_digits(probabilities)

        blank_count = min(int(self.hyperparameters["answer_anchor_blanks"]), len(probabilities))
        backward_rank = list(range(len(probabilities) - 1, -1, -1))
        blank_positions = backward_rank[:blank_count]
        fill_positions = blank_positions if self.fill_order == "backward" else sorted(blank_positions)

        fixed_digits = {index: digit for index, digit in enumerate(first_pass_digits) if index not in blank_positions}
        digits, best_answer, fill_consistency, posterior_value = select_best_answer_constrained_assignment(
            example=example,
            probabilities=probabilities,
            hyperparameters=self.hyperparameters,
            selected_positions=fill_positions,
            fixed_digits=fixed_digits,
            unrestricted_answer_log_masses=unrestricted_answer_log_masses,
        )

        return build_prediction_record(
            example=example,
            probabilities=probabilities,
            hyperparameters=self.hyperparameters,
            budget_regime=self.budget_regime,
            digits=digits,
            selected_positions=blank_positions,
            repair_ratio=float(len(blank_positions) / max(len(probabilities), 1)),
            token_count=base_reasoning_tokens(example.task_type, self.hyperparameters)
            + len(blank_positions)
            + int(self.hyperparameters["answer_candidates"]),
            blank_fill_consistency=fill_consistency,
            answer_posterior_value=posterior_value if best_answer is not None else 0.0,
        )

class ForwardOrderClozeInsteadOfBackwardCloze(AnswerAnchoredBackwardClozeReconstruction):
    name = "ForwardOrderClozeInsteadOfBackwardCloze"
    budget_regime = "tight_budget"

    def __init__(self, hyperparameters: dict, seed: int = 0) -> None:
        super().__init__(hyperparameters=hyperparameters, seed=seed, fill_order="forward")

class UncertaintyTargetedSelectiveMaskedRepair(BaseCondition):
    name = "UncertaintyTargetedSelectiveMaskedRepair"
    budget_regime = "moderate_budget"

    def _position_scores(self, example, probabilities: np.ndarray) -> np.ndarray:
        entropies = compute_entropies(probabilities)
        first_pass_digits = argmax_digits(probabilities)
        smoothed_entropy = moving_average(entropies, int(self.hyperparameters["uncertainty_smoothing_window"]))
        scores = []
        for index, probability in enumerate(probabilities):
            margin_penalty = 1.0 - top2_margin(probability)
            flip_risk = answer_flip_risk(
                example=example,
                probabilities=probabilities,
                digits=first_pass_digits,
                position=index,
                hyperparameters=self.hyperparameters,
            )
            scores.append(float(smoothed_entropy[index] * (1.0 + 1.5 * flip_risk) * (0.5 + margin_penalty)))
        return np.asarray(scores, dtype=float)

    def _selected_positions(self, example, probabilities: np.ndarray) -> list[int]:
        scores = self._position_scores(example, probabilities)
        max_positions = min(
            int(self.hyperparameters["selective_max_blank_spans"]),
            max(1, math.ceil(float(self.hyperparameters["selective_max_masked_fraction"]) * len(probabilities))),
        )
        selected = [int(index) for index in np.argsort(scores)[-max_positions:][::-1]]
        return sorted(selected)

    def predict(self, example, probabilities: np.ndarray) -> dict:
        first_pass_digits = argmax_digits(probabilities)
        selected_positions = self._selected_positions(example, probabilities)
        fixed_digits = {
            index: digit
            for index, digit in enumerate(first_pass_digits)
            if index not in selected_positions
        }

        unrestricted_answer_log_masses, _, _ = aggregate_answer_statistics(
            example=example,
            probabilities=probabilities,
            hyperparameters=self.hyperparameters,
            fixed_digits=None,
        )
        digits, best_answer, fill_consistency, posterior_value = select_best_answer_constrained_assignment(
            example=example,
            probabilities=probabilities,
            hyperparameters=self.hyperparameters,
            selected_positions=selected_positions,
            fixed_digits=fixed_digits,
            unrestricted_answer_log_masses=unrestricted_answer_log_masses,
        )

        return build_prediction_record(
            example=example,
            probabilities=probabilities,
            hyperparameters=self.hyperparameters,
            budget_regime=self.budget_regime,
            digits=digits,
            selected_positions=selected_positions,
            repair_ratio=float(len(selected_positions) / max(len(probabilities), 1)),
            token_count=base_reasoning_tokens(example.task_type, self.hyperparameters)
            + (2 * len(selected_positions))
            + int(bool(self.hyperparameters["answer_verification"])),
            blank_fill_consistency=fill_consistency,
            answer_posterior_value=posterior_value if best_answer is not None else 0.0,
        )

class RandomSpanMaskedRepair(UncertaintyTargetedSelectiveMaskedRepair):
    name = "RandomSpanMaskedRepair"
    budget_regime = "moderate_budget"

    def _selected_positions(self, example, probabilities: np.ndarray) -> list[int]:
        count = min(
            int(self.hyperparameters["selective_max_blank_spans"]),
            max(1, math.ceil(float(self.hyperparameters["selective_max_masked_fraction"]) * len(probabilities))),
        )
        rng = np.random.default_rng(stable_int_seed(self.name, self.seed, example.example_id, len(probabilities)))
        selected = rng.choice(len(probabilities), size=count, replace=False)
        return sorted(int(position) for position in selected.tolist())

class WholeRationaleMaskRepair(UncertaintyTargetedSelectiveMaskedRepair):
    name = "WholeRationaleMaskRepair"
    budget_regime = "moderate_budget"

    def _selected_positions(self, example, probabilities: np.ndarray) -> list[int]:
        return list(range(len(probabilities)))

def build_condition_registry(hyperparameters: dict) -> dict[str, callable]:
    return {
        "StandardLeftToRightChainOfThought": lambda seed: StandardLeftToRightChainOfThought(hyperparameters, seed),
        "BudgetMatchedSelfConsistencyConsensus": lambda seed: BudgetMatchedSelfConsistencyConsensus(hyperparameters, seed),
        "AnswerAnchoredBackwardClozeReconstruction": lambda seed: AnswerAnchoredBackwardClozeReconstruction(
            hyperparameters, seed
        ),
        "UncertaintyTargetedSelectiveMaskedRepair": lambda seed: UncertaintyTargetedSelectiveMaskedRepair(
            hyperparameters, seed
        ),
        "RandomSpanMaskedRepair": lambda seed: RandomSpanMaskedRepair(hyperparameters, seed),
        "ForwardOrderClozeInsteadOfBackwardCloze": lambda seed: ForwardOrderClozeInsteadOfBackwardCloze(
            hyperparameters, seed
        ),
        "WholeRationaleMaskRepair": lambda seed: WholeRationaleMaskRepair(hyperparameters, seed),
    }