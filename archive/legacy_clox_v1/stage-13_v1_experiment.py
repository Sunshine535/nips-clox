from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch

from data import (
    build_evaluation_examples,
    load_digit_data,
    precompute_example_probabilities,
    train_digit_model,
)
from methods import build_condition_registry
from utils import (
    bootstrap_ci,
    build_harness,
    cohen_d,
    mean_std,
    paired_bootstrap_ci,
    prepare_runtime_environment,
    save_json,
)

HYPERPARAMETERS: dict[str, Any] = {
    "time_budget_seconds": 120.0,
    "stop_fraction": 0.85,
    "data_root": "./data",
    "synthetic_train_size": 4000,
    "synthetic_test_size": 1200,
    "train_subset_size": 2000,
    "val_subset_size": 500,
    "batch_size": 64,
    "eval_batch_size": 256,
    "num_epochs": 8,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "hidden_dim_1": 256,
    "hidden_dim_2": 128,
    "dropout": 0.10,
    "easy_noise_std": 0.05,
    "medium_noise_std": 0.10,
    "hard_noise_std": 0.18,
    "medium_occlusion_size": 4,
    "hard_occlusion_size": 8,
    "local_sequence_length": 4,
    "nonlocal_sequence_length": 5,
    "nonlocal_threshold": 3,
    "eval_local_examples": 40,
    "eval_nonlocal_examples": 40,
    "local_easy_labels": [0, 1, 2, 3, 4],
    "local_hard_labels": [5, 6, 7, 8, 9],
    "beam_topk": 3,
    "beam_size": 32,
    "answer_candidates": 4,
    "answer_anchor_blanks": 2,
    "self_consistency_samples": 5,
    "temperature_self_consistency": 0.8,
    "uncertainty_smoothing_window": 3,
    "selective_max_blank_spans": 2,
    "selective_max_masked_fraction": 0.4,
    "answer_verification": True,
    "score_normalization_alpha": 0.5,
    "eval_subset_seed": 123,
    "seeds": [0, 1, 2],
    "primary_condition": "UncertaintyTargetedSelectiveMaskedRepair",
}

def load_plan_and_guidance() -> tuple[dict[str, Any], dict[str, Any] | None]:
    default_plan = {
        "condition_order": [
            "StandardLeftToRightChainOfThought",
            "BudgetMatchedSelfConsistencyConsensus",
            "AnswerAnchoredBackwardClozeReconstruction",
            "ForwardOrderClozeInsteadOfBackwardCloze",
            "UncertaintyTargetedSelectiveMaskedRepair",
            "RandomSpanMaskedRepair",
            "WholeRationaleMaskRepair",
        ],
        "metrics": {
            "primary_metric": {"name": "final_answer_error_rate"},
        },
    }

    plan = default_plan
    guidance = None

    plan_path = os.path.join(os.getcwd(), "plan.json")
    if os.path.exists(plan_path):
        try:
            with open(plan_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                plan = loaded
        except Exception:
            pass

    guidance_path = os.path.join(os.getcwd(), "guidance.json")
    if os.path.exists(guidance_path):
        try:
            with open(guidance_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                guidance = loaded
        except Exception:
            guidance = None

    return plan, guidance

def resolve_condition_order(plan: dict[str, Any], condition_registry: dict[str, Any]) -> list[str]:
    requested = plan.get("condition_order", [])
    if not isinstance(requested, list):
        requested = []
    filtered = [str(name) for name in requested if str(name) in condition_registry]
    if filtered:
        return filtered
    return list(condition_registry.keys())

def estimate_runtime_seconds(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    eval_examples,
    device: torch.device,
    condition_order: list[str],
    condition_registry: dict[str, Any],
) -> float:
    del train_labels, device, condition_registry
    train_size = float(len(train_images))
    eval_items = float(sum(int(example.images.shape[0]) for example in eval_examples))
    condition_factor = float(len(condition_order))
    seed_factor = float(len(HYPERPARAMETERS["seeds"]))
    estimate = 5.0 + 0.003 * train_size + 0.02 * eval_items * max(condition_factor, 1.0) * max(seed_factor, 1.0)
    return float(estimate)

def split_confidence_bins(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"threshold": 0.5}
    confidences = np.asarray([float(record.get("confidence", 0.0)) for record in records], dtype=float)
    confidences = confidences[np.isfinite(confidences)]
    if confidences.size == 0:
        return {"threshold": 0.5}
    return {"threshold": float(np.median(confidences))}

def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())

def summarize_seed_records(
    condition_name: str,
    seed: int,
    records: list[dict[str, Any]],
    confidence_bins: dict[str, float] | None,
    expected_examples: int,
    baseline_error_rate: float | None,
) -> dict[str, Any] | None:
    del condition_name, seed
    if not records:
        return None

    threshold = float(confidence_bins["threshold"]) if confidence_bins and "threshold" in confidence_bins else 0.5

    exact_matches = [float(record["correct"]) for record in records]
    errors = [1.0 - float(record["correct"]) for record in records]
    low_conf_errors = [
        1.0 - float(record["correct"]) for record in records if float(record.get("confidence", 0.0)) < threshold
    ]
    high_conf_errors = [
        1.0 - float(record["correct"]) for record in records if float(record.get("confidence", 0.0)) >= threshold
    ]
    local_errors = [
        1.0 - float(record["correct"]) for record in records if str(record.get("task_type")) == "local_recoverable"
    ]
    nonlocal_errors = [
        1.0 - float(record["correct"]) for record in records if str(record.get("task_type")) != "local_recoverable"
    ]

    coverage = float(len(records) / max(int(expected_examples), 1))
    final_answer_error_rate = _safe_mean(errors)

    summary = {
        "num_records": int(len(records)),
        "expected_examples": int(expected_examples),
        "coverage": coverage,
        "exact_match_accuracy": _safe_mean(exact_matches),
        "final_answer_error_rate": final_answer_error_rate,
        "local_recoverable_subset_error_rate": _safe_mean(local_errors),
        "non_local_subset_error_rate": _safe_mean(nonlocal_errors),
        "low_confidence_error_rate": _safe_mean(low_conf_errors),
        "high_confidence_error_rate": _safe_mean(high_conf_errors),
        "mean_confidence": _safe_mean([float(record.get("confidence", 0.0)) for record in records]),
        "mean_answer_posterior": _safe_mean([float(record.get("answer_posterior", 0.0)) for record in records]),
        "mean_tokens": _safe_mean([float(record.get("tokens", 0.0)) for record in records]),
        "mean_latency_ms": _safe_mean([float(record.get("latency_ms", 0.0)) for record in records]),
        "mean_repair_ratio": _safe_mean([float(record.get("repair_ratio", 0.0)) for record in records]),
        "mean_blank_fill_consistency": _safe_mean(
            [float(record.get("blank_fill_consistency", 0.0)) for record in records]
        ),
        "mean_sensitivity": _safe_mean([float(record.get("sensitivity", 0.0)) for record in records]),
        "mean_causal_blank_sensitivity_gap": _safe_mean(
            [float(record.get("causal_blank_sensitivity_gap", 0.0)) for record in records]
        ),
    }

    if baseline_error_rate is not None and np.isfinite(float(baseline_error_rate)):
        summary["delta_final_answer_error_rate_vs_standard"] = float(final_answer_error_rate - float(baseline_error_rate))
    else:
        summary["delta_final_answer_error_rate_vs_standard"] = 0.0

    return summary

def aggregate_condition_summaries(condition_name: str, seed_summaries: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    del condition_name
    if not seed_summaries:
        return None

    metrics = [
        "final_answer_error_rate",
        "exact_match_accuracy",
        "local_recoverable_subset_error_rate",
        "non_local_subset_error_rate",
        "low_confidence_error_rate",
        "high_confidence_error_rate",
        "mean_confidence",
        "mean_answer_posterior",
        "mean_tokens",
        "mean_latency_ms",
        "mean_repair_ratio",
        "mean_blank_fill_consistency",
        "mean_sensitivity",
        "mean_causal_blank_sensitivity_gap",
        "delta_final_answer_error_rate_vs_standard",
    ]

    aggregate: dict[str, Any] = {
        "num_seeds": int(len(seed_summaries)),
        "seeds": sorted(int(seed) for seed in seed_summaries),
    }

    for metric in metrics:
        values = [float(summary.get(metric, float("nan"))) for summary in seed_summaries.values()]
        mean_value, std_value = mean_std(values)
        ci_low, ci_high = bootstrap_ci(values, num_samples=200, seed=0)
        aggregate[f"{metric}_mean"] = mean_value
        aggregate[f"{metric}_std"] = std_value
        aggregate[f"{metric}_ci_low"] = ci_low
        aggregate[f"{metric}_ci_high"] = ci_high

    return aggregate

def build_pairwise_comparisons(per_example_results: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    condition_names = list(per_example_results.keys())
    comparisons: dict[str, Any] = {}

    for i, left_name in enumerate(condition_names):
        for right_name in condition_names[i + 1 :]:
            left_errors = []
            right_errors = []

            common_seeds = sorted(set(per_example_results.get(left_name, {})) & set(per_example_results.get(right_name, {})))
            for seed in common_seeds:
                left_records = {
                    str(record["example_id"]): record for record in per_example_results[left_name].get(seed, [])
                }
                right_records = {
                    str(record["example_id"]): record for record in per_example_results[right_name].get(seed, [])
                }
                common_examples = sorted(set(left_records) & set(right_records))
                for example_id in common_examples:
                    del example_id
                    left_errors.append(1.0 - float(left_records[next(iter([k for k in left_records if k in right_records]))]["correct"]))
                    right_errors.append(1.0 - float(right_records[next(iter([k for k in right_records if k in left_records]))]["correct"]))

            pair_key = f"{left_name}__vs__{right_name}"
            if not left_errors or not right_errors:
                comparisons[pair_key] = {
                    "num_paired_examples": 0,
                    "mean_error_difference_left_minus_right": float("nan"),
                    "paired_ci_low": float("nan"),
                    "paired_ci_high": float("nan"),
                    "cohen_d": float("nan"),
                }
                continue

            diffs = np.asarray(left_errors, dtype=float) - np.asarray(right_errors, dtype=float)
            ci_low, ci_high = paired_bootstrap_ci(left_errors, right_errors, num_samples=200, seed=0)
            comparisons[pair_key] = {
                "num_paired_examples": int(len(diffs)),
                "mean_error_difference_left_minus_right": float(np.mean(diffs)),
                "paired_ci_low": float(ci_low),
                "paired_ci_high": float(ci_high),
                "cohen_d": float(cohen_d(left_errors, right_errors)),
            }

    return comparisons

def main() -> None:
    prepare_runtime_environment(os.getcwd())
    plan, guidance = load_plan_and_guidance()
    harness = build_harness(
        time_budget=float(HYPERPARAMETERS["time_budget_seconds"]),
        stop_fraction=float(HYPERPARAMETERS["stop_fraction"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using_device: {device}")
    print(f"SEED_INFO: using {len(HYPERPARAMETERS['seeds'])} seeds")

    train_images, train_labels, test_images, test_labels, dataset_source = load_digit_data(
        data_root=str(HYPERPARAMETERS["data_root"]),
        synthetic_train_size=int(HYPERPARAMETERS["synthetic_train_size"]),
        synthetic_test_size=int(HYPERPARAMETERS["synthetic_test_size"]),
    )
    eval_examples = build_evaluation_examples(
        test_images=test_images,
        test_labels=test_labels,
        hyperparameters=HYPERPARAMETERS,
        seed=int(HYPERPARAMETERS["eval_subset_seed"]),
    )

    condition_registry = build_condition_registry(HYPERPARAMETERS)
    condition_order = resolve_condition_order(plan, condition_registry)
    print(f"dataset_source: {dataset_source}")
    print(f"examples_total: {len(eval_examples)}")
    print(f"conditions: {', '.join(condition_order)}")

    runtime_estimate = estimate_runtime_seconds(
        train_images=train_images,
        train_labels=train_labels,
        eval_examples=eval_examples,
        device=device,
        condition_order=condition_order,
        condition_registry=condition_registry,
    )
    print(f"TIME_ESTIMATE: {runtime_estimate:.1f}s")

    all_seed_summaries = {condition_name: {} for condition_name in condition_order}
    per_example_results = {condition_name: {} for condition_name in condition_order}
    training_summaries: dict[int, dict[str, Any]] = {}
    stopped_early = False
    failure_message = None
    wall_start = time.perf_counter()
    primary_metric_name = plan.get("metrics", {}).get("primary_metric", {}).get("name", "final_answer_error_rate")

    for seed in HYPERPARAMETERS["seeds"]:
        if harness.should_stop():
            print(f"STOP: time guard triggered before seed={seed}")
            stopped_early = True
            break

        try:
            model, training_summary = train_digit_model(
                train_images=train_images,
                train_labels=train_labels,
                hyperparameters=HYPERPARAMETERS,
                seed=int(seed),
                device=device,
                harness=harness,
            )
        except RuntimeError as exc:
            failure_message = str(exc)
            stopped_early = True
            break

        training_summaries[int(seed)] = training_summary
        probability_map = precompute_example_probabilities(
            model=model,
            examples=eval_examples,
            device=device,
            batch_size=int(HYPERPARAMETERS["eval_batch_size"]),
        )

        baseline_error_rate = None
        confidence_bins = None
        standard_record_map = None

        for condition_name in condition_order:
            if harness.should_stop():
                print(f"STOP: time guard triggered before condition={condition_name} seed={seed}")
                stopped_early = True
                break

            condition = condition_registry[condition_name](int(seed))
            records = []

            for example in eval_examples:
                if harness.should_stop():
                    print(f"STOP: time guard triggered during condition={condition_name} seed={seed}")
                    stopped_early = True
                    break

                start = time.perf_counter()
                record = condition.predict(example, probability_map[example.example_id])
                latency_ms = (time.perf_counter() - start) * 1000.0
                record["latency_ms"] = float(latency_ms)
                record["example_id"] = example.example_id
                record["ground_truth"] = int(example.answer)
                record["task_type"] = example.task_type
                record["condition"] = condition_name
                record["seed"] = int(seed)
                record["corruption_levels"] = list(example.corruption_levels)

                numeric_fields = [
                    "confidence",
                    "answer_posterior",
                    "tokens",
                    "repair_ratio",
                    "blank_fill_consistency",
                    "sensitivity",
                    "latency_ms",
                ]
                if not all(harness.check_value(record.get(field, float("nan")), field) for field in numeric_fields):
                    print("SKIP: NaN/Inf detected")
                    continue
                records.append(record)

            if standard_record_map is not None:
                for record in records:
                    matched_k = min(2, max(1, len(record.get("selected_positions", []))))
                    baseline_record = standard_record_map.get(record["example_id"])
                    if baseline_record is None:
                        record["causal_blank_sensitivity_gap"] = 0.0
                    else:
                        baseline_sensitivity = float(
                            baseline_record.get("standard_sensitivity_by_k", {}).get(matched_k, 0.0)
                        )
                        record["causal_blank_sensitivity_gap"] = float(record["sensitivity"] - baseline_sensitivity)
            else:
                for record in records:
                    record["causal_blank_sensitivity_gap"] = 0.0

            expected_examples = len(eval_examples)
            summary = summarize_seed_records(
                condition_name=condition_name,
                seed=int(seed),
                records=records,
                confidence_bins=confidence_bins,
                expected_examples=expected_examples,
                baseline_error_rate=baseline_error_rate,
            )
            if summary is None:
                continue

            all_seed_summaries[condition_name][int(seed)] = summary
            per_example_results[condition_name][int(seed)] = records

            if condition_name == "StandardLeftToRightChainOfThought":
                confidence_bins = split_confidence_bins(records)
                standard_record_map = {record["example_id"]: record for record in records}
                baseline_error_rate = summary["final_answer_error_rate"]

            print(f"condition={condition_name} seed={seed} final_answer_error_rate: {summary['final_answer_error_rate']:.4f}")
            print(
                f"condition={condition_name} seed={seed} "
                f"local_recoverable_subset_error_rate: {summary['local_recoverable_subset_error_rate']:.4f} "
                f"non_local_subset_error_rate: {summary['non_local_subset_error_rate']:.4f}"
            )
            print(
                f"condition={condition_name} seed={seed} "
                f"low_confidence_error_rate: {summary['low_confidence_error_rate']:.4f} "
                f"high_confidence_error_rate: {summary['high_confidence_error_rate']:.4f}"
            )
            harness.report_metric(
                f"{condition_name}_seed_{seed}_{primary_metric_name}",
                float(summary["final_answer_error_rate"]),
            )

            if stopped_early:
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if stopped_early:
            break

    aggregate_results = {}
    for condition_name in condition_order:
        aggregate = aggregate_condition_summaries(condition_name, all_seed_summaries.get(condition_name, {}))
        if aggregate is None:
            continue
        aggregate_results[condition_name] = aggregate
        print(
            f"condition={condition_name} "
            f"final_answer_error_rate_mean: {aggregate['final_answer_error_rate_mean']:.4f} "
            f"final_answer_error_rate_std: {aggregate['final_answer_error_rate_std']:.4f}"
        )
        print(
            f"condition={condition_name} "
            f"exact_match_accuracy_mean: {aggregate['exact_match_accuracy_mean']:.4f} "
            f"exact_match_accuracy_std: {aggregate['exact_match_accuracy_std']:.4f}"
        )

    comparisons = build_pairwise_comparisons(per_example_results)

    primary_condition = str(HYPERPARAMETERS["primary_condition"])
    if primary_condition not in aggregate_results and aggregate_results:
        primary_condition = min(
            aggregate_results,
            key=lambda name: aggregate_results[name]["final_answer_error_rate_mean"],
        )

    if primary_condition in aggregate_results:
        primary_mean = float(aggregate_results[primary_condition]["final_answer_error_rate_mean"])
        primary_std = float(aggregate_results[primary_condition]["final_answer_error_rate_std"])
    else:
        primary_mean = float("nan")
        primary_std = float("nan")

    best_condition = None
    if aggregate_results:
        best_condition = min(
            aggregate_results,
            key=lambda name: aggregate_results[name]["final_answer_error_rate_mean"],
        )

    results_payload = {
        "hyperparameters": HYPERPARAMETERS,
        "dataset_source": dataset_source,
        "condition_order": condition_order,
        "implemented_not_run_conditions": sorted(set(condition_registry) - set(condition_order)),
        "training_summaries": training_summaries,
        "seed_summaries": all_seed_summaries,
        "aggregate_results": aggregate_results,
        "per_example_results": per_example_results,
        "pairwise_comparisons": comparisons,
        "primary_metric_name": primary_metric_name,
        "primary_metric_condition": primary_condition,
        "primary_metric_mean": primary_mean,
        "primary_metric_std": primary_std,
        "best_condition_by_primary_metric": best_condition,
        "time_elapsed_seconds": float(time.perf_counter() - wall_start),
        "stopped_early": bool(stopped_early),
        "failure_message": failure_message,
        "guidance_loaded": bool(guidance),
    }

    if np.isfinite(primary_mean):
        harness.report_metric(primary_metric_name, primary_mean)
        harness.report_metric("primary_metric", primary_mean)

    try:
        harness.finalize()
    except Exception:
        pass

    save_json(os.path.join(os.getcwd(), "results.json"), results_payload)
    if np.isfinite(primary_mean):
        print(f"primary_metric: {primary_mean:.4f} ± {primary_std:.4f}")
    else:
        print("primary_metric: nan")

if __name__ == "__main__":
    main()