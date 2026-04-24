from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from methods import build_condition_registry
from utils import (
    bootstrap_ci,
    build_harness,
    mean_std,
    paired_bootstrap_ci,
    prepare_runtime_environment,
    save_json,
    set_all_seeds,
)

HYPERPARAMETERS: dict[str, Any] = {
    "data_root": "./data",
    "synthetic_train_size": 4000,
    "synthetic_test_size": 1200,
    "train_subset_size": 2048,
    "val_subset_size": 512,
    "batch_size": 64,
    "eval_batch_size": 128,
    "num_epochs": 8,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "hidden_dim_1": 256,
    "hidden_dim_2": 128,
    "dropout": 0.1,
    "early_stopping_patience": 3,
    "label_smoothing": 0.02,
    "train_noise_std": 0.03,
    "easy_noise_std": 0.03,
    "medium_noise_std": 0.08,
    "hard_noise_std": 0.14,
    "medium_occlusion_size": 4,
    "hard_occlusion_size": 8,
    "local_sequence_length": 4,
    "nonlocal_sequence_length": 5,
    "nonlocal_threshold": 3,
    "local_easy_labels": [0, 1, 2, 3, 7],
    "local_hard_labels": [4, 5, 6, 8, 9],
    "eval_local_examples": 24,
    "eval_nonlocal_examples": 24,
    "eval_subset_seed": 123,
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
    "time_budget_seconds": 60.0,
    "stop_fraction": 0.85,
    "seeds": [0, 1],
    "primary_condition": "UncertaintyTargetedSelectiveMaskedRepair",
}

@dataclass
class TaskExample:
    example_id: str
    task_type: str
    images: torch.Tensor
    labels: list[int]
    answer: int
    corruption_levels: list[str]

_SEGMENTS = {
    0: (0, 1, 2, 4, 5, 6),
    1: (2, 5),
    2: (0, 2, 3, 4, 6),
    3: (0, 2, 3, 5, 6),
    4: (1, 2, 3, 5),
    5: (0, 1, 3, 5, 6),
    6: (0, 1, 3, 4, 5, 6),
    7: (0, 2, 5),
    8: (0, 1, 2, 3, 4, 5, 6),
    9: (0, 1, 2, 3, 5, 6),
}

_SEGMENT_RECTS = {
    0: (3, 2, 25, 5),
    1: (2, 4, 5, 14),
    2: (23, 4, 26, 14),
    3: (3, 12, 25, 15),
    4: (2, 14, 5, 24),
    5: (23, 14, 26, 24),
    6: (3, 23, 25, 26),
}

class DigitPerceptionMLP(nn.Module):
    def __init__(self, hyperparameters: dict) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, int(hyperparameters["hidden_dim_1"])),
            nn.GELU(),
            nn.Dropout(float(hyperparameters["dropout"])),
            nn.Linear(int(hyperparameters["hidden_dim_1"]), int(hyperparameters["hidden_dim_2"])),
            nn.GELU(),
            nn.Dropout(float(hyperparameters["dropout"]) * 0.5),
            nn.Linear(int(hyperparameters["hidden_dim_2"]), 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images.float())

def answer_from_digits(task_type: str, digits, hyperparameters: dict) -> int:
    digits = [int(value) for value in digits]
    if task_type == "local_recoverable":
        return int(sum(digits))
    ordered = sorted(digits)
    threshold = int(hyperparameters["nonlocal_threshold"])
    return int(ordered[len(ordered) // 2] if sum(digit >= 5 for digit in digits) >= threshold else max(digits))

def base_reasoning_tokens(task_type: str, hyperparameters: dict) -> int:
    if task_type == "local_recoverable":
        return int(hyperparameters["local_sequence_length"]) + 3
    return int(hyperparameters["nonlocal_sequence_length"]) + 5

def _render_synthetic_digit(label: int, rng: np.random.Generator) -> torch.Tensor:
    canvas = torch.zeros(1, 28, 28, dtype=torch.float32)
    for segment_id in _SEGMENTS[int(label)]:
        x0, y0, x1, y1 = _SEGMENT_RECTS[segment_id]
        dx = int(rng.integers(-1, 2))
        dy = int(rng.integers(-1, 2))
        x0 = max(0, min(27, x0 + dx))
        x1 = max(x0 + 1, min(28, x1 + dx))
        y0 = max(0, min(27, y0 + dy))
        y1 = max(y0 + 1, min(28, y1 + dy))
        canvas[:, y0:y1, x0:x1] = 1.0
    canvas = torch.roll(
        canvas,
        shifts=(int(rng.integers(-1, 2)), int(rng.integers(-1, 2))),
        dims=(1, 2),
    )
    noise = torch.from_numpy(rng.normal(0.0, 0.04, size=canvas.shape)).to(torch.float32)
    return torch.clamp(canvas + noise, 0.0, 1.0)

def _build_synthetic_dataset(size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for _ in range(max(int(size), 1)):
        label = int(rng.integers(0, 10))
        images.append(_render_synthetic_digit(label, rng))
        labels.append(label)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

def load_digit_data(
    data_root: str,
    synthetic_train_size: int,
    synthetic_test_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    candidate_roots = []
    if data_root:
        candidate_roots.append(str(data_root))
    if "/opt/datasets" not in candidate_roots:
        candidate_roots.append("/opt/datasets")

    for root in candidate_roots:
        try:
            from torchvision import datasets

            train_dataset = datasets.MNIST(root=root, train=True, download=False)
            test_dataset = datasets.MNIST(root=root, train=False, download=False)
            train_images = train_dataset.data.unsqueeze(1).to(torch.float32) / 255.0
            test_images = test_dataset.data.unsqueeze(1).to(torch.float32) / 255.0
            train_labels = train_dataset.targets.long()
            test_labels = test_dataset.targets.long()
            return train_images, train_labels, test_images, test_labels, f"mnist_cached:{root}"
        except Exception:
            continue

    train_images, train_labels = _build_synthetic_dataset(synthetic_train_size, seed=0)
    test_images, test_labels = _build_synthetic_dataset(synthetic_test_size, seed=1)
    return train_images, train_labels, test_images, test_labels, "synthetic_seven_segment"

def build_train_val_loaders(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    hyperparameters: dict,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    requested = min(
        len(train_images),
        int(hyperparameters["train_subset_size"]) + int(hyperparameters["val_subset_size"]),
    )
    if requested < 2:
        raise RuntimeError("insufficient data for train/val split")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_images))[:requested]
    train_count = max(1, min(int(hyperparameters["train_subset_size"]), requested - 1))
    val_count = max(1, requested - train_count)
    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_images[train_indices], train_labels[train_indices]),
        batch_size=max(1, int(hyperparameters["batch_size"])),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(train_images[val_indices], train_labels[val_indices]),
        batch_size=max(1, int(hyperparameters["batch_size"])),
        shuffle=False,
    )
    return train_loader, val_loader

def evaluate_digit_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in data_loader:
            logits = model(images.to(device=device, dtype=torch.float32))
            predictions = logits.argmax(dim=-1).cpu()
            correct += int((predictions == labels).sum())
            total += int(labels.numel())
    return float(correct / max(total, 1))

def train_digit_model(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    hyperparameters: dict,
    seed: int,
    device: torch.device,
    harness,
) -> tuple[nn.Module, dict]:
    set_all_seeds(seed)
    model = DigitPerceptionMLP(hyperparameters).to(device)
    train_loader, val_loader = build_train_val_loaders(train_images, train_labels, hyperparameters, seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=min(float(hyperparameters["learning_rate"]), 3e-4),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(hyperparameters["num_epochs"]), 1),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    history = []
    epochs_completed = 0
    patience = max(1, int(hyperparameters.get("early_stopping_patience", 3)))
    stale_epochs = 0
    label_smoothing = float(hyperparameters.get("label_smoothing", 0.02))
    train_noise_std = float(hyperparameters.get("train_noise_std", 0.03))

    for epoch in range(int(hyperparameters["num_epochs"])):
        if harness.should_stop():
            break

        model.train()
        epoch_losses = []
        epoch_grad_norms = []

        for images, labels in train_loader:
            if harness.should_stop():
                break

            optimizer.zero_grad(set_to_none=True)
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device)

            if train_noise_std > 0.0:
                images = torch.clamp(images + torch.randn_like(images) * train_noise_std, 0.0, 1.0)

            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

            if torch.isnan(loss):
                continue
            loss_value = float(loss.item())
            if (not torch.isfinite(loss)) or loss_value > 100.0:
                raise RuntimeError("training diverged")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(hyperparameters.get("gradient_clip_norm", 1.0)),
            )
            grad_norm_value = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            if not np.isfinite(grad_norm_value):
                raise RuntimeError("non-finite gradient norm")

            optimizer.step()
            epoch_losses.append(loss_value)
            epoch_grad_norms.append(grad_norm_value)

        scheduler.step()
        val_accuracy = evaluate_digit_accuracy(model, val_loader, device)
        epochs_completed += 1
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else float("nan")
        learning_rate = float(optimizer.param_groups[0]["lr"])
        print(
            f"train seed={seed} epoch={epoch} "
            f"train_loss: {train_loss:.4f} grad_norm: {grad_norm:.4f} "
            f"lr: {learning_rate:.6f} val_digit_accuracy: {val_accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "grad_norm": grad_norm,
                "learning_rate": learning_rate,
                "val_digit_accuracy": val_accuracy,
            }
        )

        improvement = val_accuracy - best_val_accuracy
        if improvement >= 1e-4:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break
        if harness.should_stop():
            break

    model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "epochs_completed": epochs_completed,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
    }

def _apply_corruption(
    image: torch.Tensor,
    level: str,
    rng: np.random.Generator,
    hyperparameters: dict,
) -> torch.Tensor:
    corrupted = image.clone().to(torch.float32)
    if level == "easy":
        noise_std = float(hyperparameters["easy_noise_std"])
        occlusion_size = 0
    elif level == "medium":
        noise_std = float(hyperparameters["medium_noise_std"])
        occlusion_size = int(hyperparameters["medium_occlusion_size"])
    else:
        noise_std = float(hyperparameters["hard_noise_std"])
        occlusion_size = int(hyperparameters["hard_occlusion_size"])

    corrupted = torch.clamp(
        corrupted + torch.from_numpy(rng.normal(0.0, noise_std, size=corrupted.shape)).to(torch.float32),
        0.0,
        1.0,
    )

    if occlusion_size > 0:
        occlusion_size = max(1, min(occlusion_size, 28))
        y0 = int(rng.integers(0, 28 - occlusion_size + 1))
        x0 = int(rng.integers(0, 28 - occlusion_size + 1))
        factor = 0.0 if level == "hard" else 0.35
        corrupted[:, y0 : y0 + occlusion_size, x0 : x0 + occlusion_size] *= factor

    if level == "hard":
        corrupted = torch.roll(
            corrupted,
            shifts=(int(rng.integers(-2, 3)), int(rng.integers(-2, 3))),
            dims=(1, 2),
        )

    return torch.clamp(corrupted, 0.0, 1.0)

def build_evaluation_examples(
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    hyperparameters: dict,
    seed: int,
) -> list[TaskExample]:
    rng = np.random.default_rng(seed)
    label_buckets = {label: torch.where(test_labels == label)[0].tolist() for label in range(10)}
    for bucket in label_buckets.values():
        rng.shuffle(bucket)
    pointers = {label: 0 for label in range(10)}

    easy_labels = [int(value) for value in hyperparameters["local_easy_labels"]]
    hard_labels = [int(value) for value in hyperparameters["local_hard_labels"]]

    def draw_image_for_label(label: int) -> torch.Tensor:
        bucket = label_buckets[label]
        if not bucket:
            replacement = int(rng.integers(0, len(test_images)))
            return test_images[replacement]
        pointer = pointers[label]
        if pointer >= len(bucket):
            rng.shuffle(bucket)
            pointer = 0
        selected_index = bucket[pointer]
        pointers[label] = pointer + 1
        return test_images[selected_index]

    examples: list[TaskExample] = []

    local_sequence_length = max(1, int(hyperparameters["local_sequence_length"]))
    for example_index in range(int(hyperparameters["eval_local_examples"])):
        labels = [int(rng.choice(easy_labels)) for _ in range(local_sequence_length - 1)]
        labels.append(int(rng.choice(hard_labels)))
        rng.shuffle(labels)
        hard_position = next(position for position, label in enumerate(labels) if label in hard_labels)
        corruption_levels = ["hard" if position == hard_position else "easy" for position in range(local_sequence_length)]
        images = [
            _apply_corruption(draw_image_for_label(label), level, rng, hyperparameters)
            for label, level in zip(labels, corruption_levels)
        ]
        examples.append(
            TaskExample(
                example_id=f"local_{example_index:02d}",
                task_type="local_recoverable",
                images=torch.stack(images),
                labels=labels,
                answer=answer_from_digits("local_recoverable", labels, hyperparameters),
                corruption_levels=corruption_levels,
            )
        )

    nonlocal_sequence_length = max(1, int(hyperparameters["nonlocal_sequence_length"]))
    for example_index in range(int(hyperparameters["eval_nonlocal_examples"])):
        labels = [int(rng.choice(hard_labels)) for _ in range(nonlocal_sequence_length)]
        hard_count = min(3, nonlocal_sequence_length)
        corruption_levels = ["hard"] * hard_count + ["medium"] * max(0, nonlocal_sequence_length - hard_count)
        rng.shuffle(corruption_levels)
        images = [
            _apply_corruption(draw_image_for_label(label), level, rng, hyperparameters)
            for label, level in zip(labels, corruption_levels)
        ]
        examples.append(
            TaskExample(
                example_id=f"nonlocal_{example_index:02d}",
                task_type="non_local",
                images=torch.stack(images),
                labels=labels,
                answer=answer_from_digits("non_local", labels, hyperparameters),
                corruption_levels=corruption_levels,
            )
        )

    return examples

def precompute_example_probabilities(
    model: nn.Module,
    examples: list[TaskExample],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if not examples:
        return {}

    flat_images = torch.cat([example.images for example in examples], dim=0)
    all_probabilities = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat_images), max(1, int(batch_size))):
            batch = flat_images[start : start + max(1, int(batch_size))].to(device=device, dtype=torch.float32)
            logits = model(batch)
            probabilities = F.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probabilities.append(probabilities)

    probability_matrix = np.concatenate(all_probabilities, axis=0)
    example_probability_map: dict[str, np.ndarray] = {}
    cursor = 0
    for example in examples:
        count = int(example.images.shape[0])
        example_probability_map[example.example_id] = probability_matrix[cursor : cursor + count]
        cursor += count
    return example_probability_map

def load_plan_and_guidance() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = {}
    guidance = {}
    cwd = os.getcwd()

    plan_path = os.path.join(cwd, "plan.json")
    guidance_path = os.path.join(cwd, "guidance.json")

    if os.path.exists(plan_path):
        try:
            with open(plan_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if isinstance(loaded, dict):
                    plan = loaded
        except Exception:
            plan = {}

    if os.path.exists(guidance_path):
        try:
            with open(guidance_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if isinstance(loaded, dict):
                    guidance = loaded
        except Exception:
            guidance = {}

    return plan, guidance

def resolve_condition_order(plan: dict[str, Any], condition_registry: dict[str, Any]) -> list[str]:
    requested = plan.get("conditions")
    if isinstance(requested, list):
        resolved = [str(name) for name in requested if str(name) in condition_registry]
        if resolved:
            return resolved
    return list(condition_registry.keys())

def estimate_runtime_seconds(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    eval_examples: list[TaskExample],
    device: torch.device,
    condition_order: list[str],
    condition_registry: dict[str, Any],
) -> float:
    del train_labels, device, condition_registry
    train_batches = max(
        1,
        int(np.ceil(min(len(train_images), int(HYPERPARAMETERS["train_subset_size"])) / int(HYPERPARAMETERS["batch_size"]))),
    )
    train_cost = len(HYPERPARAMETERS["seeds"]) * int(HYPERPARAMETERS["num_epochs"]) * train_batches * 0.03
    eval_cost = len(HYPERPARAMETERS["seeds"]) * len(condition_order) * len(eval_examples) * 0.01
    return float(train_cost + eval_cost)

def split_confidence_bins(records: list[dict[str, Any]]) -> dict[str, float]:
    confidences = np.asarray([float(record.get("confidence", np.nan)) for record in records], dtype=float)
    confidences = confidences[np.isfinite(confidences)]
    if confidences.size == 0:
        return {"threshold": 0.5}
    return {"threshold": float(np.median(confidences))}

def _safe_rate(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(array.mean())

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

    threshold = 0.5 if confidence_bins is None else float(confidence_bins.get("threshold", 0.5))
    exact_matches = [int(record.get("correct", 0)) for record in records]
    errors = [1 - value for value in exact_matches]

    local_records = [record for record in records if record.get("task_type") == "local_recoverable"]
    nonlocal_records = [record for record in records if record.get("task_type") == "non_local"]
    low_conf_records = [record for record in records if float(record.get("confidence", 0.0)) < threshold]
    high_conf_records = [record for record in records if float(record.get("confidence", 0.0)) >= threshold]

    summary = {
        "num_records": int(len(records)),
        "coverage": float(len(records) / max(expected_examples, 1)),
        "exact_match_accuracy": _safe_rate(exact_matches),
        "final_answer_error_rate": _safe_rate(errors),
        "local_recoverable_subset_error_rate": _safe_rate([1 - int(r.get("correct", 0)) for r in local_records]),
        "non_local_subset_error_rate": _safe_rate([1 - int(r.get("correct", 0)) for r in nonlocal_records]),
        "low_confidence_error_rate": _safe_rate([1 - int(r.get("correct", 0)) for r in low_conf_records]),
        "high_confidence_error_rate": _safe_rate([1 - int(r.get("correct", 0)) for r in high_conf_records]),
        "mean_confidence": _safe_rate([float(r.get("confidence", np.nan)) for r in records]),
        "mean_answer_posterior": _safe_rate([float(r.get("answer_posterior", np.nan)) for r in records]),
        "mean_tokens": _safe_rate([float(r.get("tokens", np.nan)) for r in records]),
        "mean_repair_ratio": _safe_rate([float(r.get("repair_ratio", np.nan)) for r in records]),
        "mean_blank_fill_consistency": _safe_rate([float(r.get("blank_fill_consistency", np.nan)) for r in records]),
        "mean_sensitivity": _safe_rate([float(r.get("sensitivity", np.nan)) for r in records]),
        "mean_latency_ms": _safe_rate([float(r.get("latency_ms", np.nan)) for r in records]),
        "causal_blank_sensitivity_gap_mean": _safe_rate(
            [float(r.get("causal_blank_sensitivity_gap", np.nan)) for r in records]
        ),
    }

    if baseline_error_rate is not None and np.isfinite(float(baseline_error_rate)):
        summary["error_rate_delta_vs_standard"] = float(summary["final_answer_error_rate"] - float(baseline_error_rate))
    else:
        summary["error_rate_delta_vs_standard"] = 0.0

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
        "mean_repair_ratio",
        "mean_blank_fill_consistency",
        "mean_sensitivity",
        "mean_latency_ms",
        "causal_blank_sensitivity_gap_mean",
        "coverage",
    ]

    aggregate: dict[str, Any] = {"num_seeds": int(len(seed_summaries))}
    for metric in metrics:
        values = [float(summary.get(metric, np.nan)) for summary in seed_summaries.values()]
        mean_value, std_value = mean_std(values)
        aggregate[f"{metric}_mean"] = mean_value
        aggregate[f"{metric}_std"] = std_value
        ci_low, ci_high = bootstrap_ci(values, seed=0)
        aggregate[f"{metric}_ci_low"] = ci_low
        aggregate[f"{metric}_ci_high"] = ci_high
    return aggregate

def build_pairwise_comparisons(per_example_results: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    condition_names = sorted(per_example_results.keys())
    comparisons: dict[str, Any] = {}

    for i, left_name in enumerate(condition_names):
        for right_name in condition_names[i + 1 :]:
            left_errors = []
            right_errors = []

            common_seeds = sorted(set(per_example_results.get(left_name, {})) & set(per_example_results.get(right_name, {})))
            for seed in common_seeds:
                left_map = {
                    str(record["example_id"]): 1 - int(record.get("correct", 0))
                    for record in per_example_results[left_name].get(seed, [])
                }
                right_map = {
                    str(record["example_id"]): 1 - int(record.get("correct", 0))
                    for record in per_example_results[right_name].get(seed, [])
                }
                common_examples = sorted(set(left_map) & set(right_map))
                for example_id in common_examples:
                    left_errors.append(float(left_map[example_id]))
                    right_errors.append(float(right_map[example_id]))

            if left_errors and right_errors:
                left_mean = float(np.mean(left_errors))
                right_mean = float(np.mean(right_errors))
                ci_low, ci_high = paired_bootstrap_ci(left_errors, right_errors, seed=0)
                comparisons[f"{left_name}__vs__{right_name}"] = {
                    "left_error_rate": left_mean,
                    "right_error_rate": right_mean,
                    "error_rate_difference_left_minus_right": float(left_mean - right_mean),
                    "difference_ci_low": ci_low,
                    "difference_ci_high": ci_high,
                    "paired_count": int(min(len(left_errors), len(right_errors))),
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