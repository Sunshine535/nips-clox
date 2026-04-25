from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy import stats


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[,\$%\.\s]+$", "", text)
    text = re.sub(r"^\s*\(?\s*", "", text)
    text = re.sub(r"\s*\)?\s*$", "", text)
    text = re.sub(r"\\boxed\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text


def check_answer_legacy_unsafe(prediction: str, reference: str, answer_type: str = "text") -> bool:
    """LEGACY checker — preserved verbatim for historical-result replay only.

    DO NOT USE for new experiments. The terminal substring fallback
    (`pred in ref or ref in pred`) silently passes contaminated answers
    (e.g. "the answer is 123 units" matches "123"). Use `check_answer`
    (delegates to strict path) or `answer_extraction.check_answer_strict`.
    """
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)

    if pred == ref:
        return True

    if answer_type in ("numeric", "math_expression"):
        try:
            pred_num = float(re.sub(r"[^\d.\-]", "", pred))
            ref_num = float(re.sub(r"[^\d.\-]", "", ref))
            return abs(pred_num - ref_num) < 1e-6
        except (ValueError, TypeError):
            pass

    if answer_type == "multiple_choice":
        _mc_patterns = [
            r"(?:the answer is|answer:)\s*\(?([a-eA-E])\)?",
            r"\(([a-eA-E])\)",
            r"(?:^|\s)([a-eA-E])\s*[\.\):]",
            r"(?:^|\s)\(?([a-eA-E])\)?\s*$",
        ]
        pred_letter = ref_letter = None
        for pat in _mc_patterns:
            if not pred_letter:
                m = re.search(pat, pred)
                if m:
                    pred_letter = m.group(1)
            if not ref_letter:
                m = re.search(pat, ref)
                if m:
                    ref_letter = m.group(1)
        if pred_letter and ref_letter:
            return pred_letter.upper() == ref_letter.upper()

    if answer_type == "boolean":
        pred_bool = pred in ("yes", "true", "1")
        ref_bool = ref in ("yes", "true", "1")
        return pred_bool == ref_bool

    return pred in ref or ref in pred


def check_answer(prediction: str, reference: str, answer_type: str = "text") -> bool:
    """Strict answer checker.

    Delegates to `answer_extraction.check_answer_strict` (no substring fallback).
    Any caller importing `evaluation.check_answer` automatically gets the strict
    path. For audit access to the unsafe historical behaviour, see
    `check_answer_legacy_unsafe`.
    """
    from answer_extraction import check_answer_strict
    return bool(check_answer_strict(prediction, reference, answer_type))


def exact_match_accuracy(
    predictions: list[str],
    references: list[str],
    answer_types: list[str] | None = None,
) -> float:
    if not predictions or not references:
        return 0.0
    types = answer_types or ["text"] * len(predictions)
    correct = sum(
        check_answer(p, r, t)
        for p, r, t in zip(predictions, references, types)
    )
    return correct / len(predictions)


def paired_bootstrap_ci(
    a_correct: list[bool],
    b_correct: list[bool],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}

    a, b = a[:n], b[:n]
    observed_diff = float(np.mean(a) - np.mean(b))

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(a[indices]) - np.mean(b[indices])

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    centered = boot_diffs - np.mean(boot_diffs)
    p_value = float(np.mean(np.abs(centered) >= abs(observed_diff)))

    return {
        "mean_diff": observed_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def mcnemar_test(a_correct: list[bool], b_correct: list[bool]) -> dict[str, float]:
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    b_only = int(np.sum(~a & b))
    a_only = int(np.sum(a & ~b))

    total_discordant = a_only + b_only
    if total_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "a_only": a_only, "b_only": b_only}

    statistic = (abs(a_only - b_only) - 1) ** 2 / total_discordant
    p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))

    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "a_only": a_only,
        "b_only": b_only,
    }


def cohens_d(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[np.isfinite(a_arr)]
    b_arr = b_arr[np.isfinite(b_arr)]

    if len(a_arr) < 2 or len(b_arr) < 2:
        return 0.0

    pooled_std = math.sqrt(
        ((len(a_arr) - 1) * np.var(a_arr, ddof=1) + (len(b_arr) - 1) * np.var(b_arr, ddof=1))
        / max(len(a_arr) + len(b_arr) - 2, 1)
    )
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[dict[str, Any]]:
    n = len(p_values)
    if n == 0:
        return []
    adjusted_alpha = alpha / n
    return [
        {
            "original_p": p,
            "adjusted_threshold": adjusted_alpha,
            "significant": p < adjusted_alpha,
            "rank": i,
        }
        for i, p in enumerate(p_values)
    ]


def compute_token_efficiency(
    accuracies: list[float],
    token_counts: list[int],
) -> dict[str, float]:
    acc = np.asarray(accuracies, dtype=float)
    tokens = np.asarray(token_counts, dtype=float)
    valid = np.isfinite(acc) & np.isfinite(tokens) & (tokens > 0)
    if not np.any(valid):
        return {"tokens_per_correct": float("inf"), "accuracy_per_1k_tokens": 0.0}

    acc, tokens = acc[valid], tokens[valid]
    correct_count = np.sum(acc > 0.5)
    total_tokens = np.sum(tokens)

    return {
        "tokens_per_correct": float(total_tokens / max(correct_count, 1)),
        "accuracy_per_1k_tokens": float(np.mean(acc) / max(np.mean(tokens) / 1000, 0.001)),
        "mean_tokens": float(np.mean(tokens)),
        "total_tokens": float(total_tokens),
    }


def per_example_win_loss_matrix(
    results: dict[str, list[bool]],
) -> dict[str, dict[str, Any]]:
    strategies = sorted(results.keys())
    matrix = {}

    for i, s_a in enumerate(strategies):
        for s_b in strategies[i + 1:]:
            a_correct = np.asarray(results[s_a], dtype=bool)
            b_correct = np.asarray(results[s_b], dtype=bool)
            n = min(len(a_correct), len(b_correct))
            a_correct, b_correct = a_correct[:n], b_correct[:n]

            both_correct = int(np.sum(a_correct & b_correct))
            a_wins = int(np.sum(a_correct & ~b_correct))
            b_wins = int(np.sum(~a_correct & b_correct))
            both_wrong = int(np.sum(~a_correct & ~b_correct))

            matrix[f"{s_a}_vs_{s_b}"] = {
                "both_correct": both_correct,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "both_wrong": both_wrong,
                "total": n,
                "a_win_rate": a_wins / max(a_wins + b_wins, 1),
            }

    return matrix


def compute_task_topology_metrics(
    per_example_results: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    all_step_counts = []
    error_propagation_pairs = []

    for strategy_name, examples in per_example_results.items():
        for ex in examples:
            steps = ex.get("step_metadata", [])
            all_step_counts.append(len(steps))

            if "logprobs" in ex and ex["logprobs"]:
                logprobs = ex["logprobs"]
                is_correct = ex.get("correct", False)
                if not is_correct and len(logprobs) > 1:
                    min_lp_idx = int(np.argmin(logprobs))
                    propagation_length = len(logprobs) - min_lp_idx
                    error_propagation_pairs.append(propagation_length)

    mean_steps = float(np.mean(all_step_counts)) if all_step_counts else 5.0
    mean_epl = float(np.mean(error_propagation_pairs)) if error_propagation_pairs else mean_steps / 2

    strategy_names = list(per_example_results.keys())
    if len(strategy_names) >= 2:
        s1, s2 = strategy_names[0], strategy_names[1]
        n = min(len(per_example_results[s1]), len(per_example_results[s2]))
        agreements = 0
        for i in range(n):
            if per_example_results[s1][i].get("correct") == per_example_results[s2][i].get("correct"):
                agreements += 1
        recoverability_proxy = agreements / max(n, 1)
    else:
        recoverability_proxy = 0.5

    return {
        "estimated_epl": mean_epl,
        "estimated_recoverability": recoverability_proxy,
        "mean_steps": mean_steps,
        "n_examples": sum(len(v) for v in per_example_results.values()),
    }
