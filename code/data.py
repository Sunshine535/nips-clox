from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import set_all_seeds


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
            nn.Linear(int(hyperparameters["hidden_dim_2"]), 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


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
    canvas = torch.zeros(1, 28, 28)
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
    noise = torch.from_numpy(rng.normal(0.0, 0.04, size=canvas.shape)).float()
    return torch.clamp(canvas + noise, 0.0, 1.0)


def _build_synthetic_dataset(size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for _ in range(int(size)):
        label = int(rng.integers(0, 10))
        images.append(_render_synthetic_digit(label, rng))
        labels.append(label)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def load_digit_data(
    data_root: str,
    synthetic_train_size: int,
    synthetic_test_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    try:
        from torchvision import datasets

        train_dataset = datasets.MNIST(root=data_root, train=True, download=False)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=False)
        train_images = train_dataset.data.unsqueeze(1).float() / 255.0
        test_images = test_dataset.data.unsqueeze(1).float() / 255.0
        train_labels = train_dataset.targets.long()
        test_labels = test_dataset.targets.long()
        return train_images, train_labels, test_images, test_labels, "mnist_cached"
    except Exception:
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
        raise RuntimeError("FAIL: NaN/divergence detected")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_images))[:requested]
    train_count = max(1, min(int(hyperparameters["train_subset_size"]), requested - 1))
    val_count = max(1, requested - train_count)
    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_images[train_indices], train_labels[train_indices]),
        batch_size=int(hyperparameters["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(train_images[val_indices], train_labels[val_indices]),
        batch_size=int(hyperparameters["batch_size"]),
        shuffle=False,
    )
    return train_loader, val_loader


def manual_sgd_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    hyperparameters: dict,
) -> tuple[float, float]:
    for parameter in model.parameters():
        parameter.grad = None

    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss_value = float(loss.item())
    if (not torch.isfinite(loss)) or loss_value > 100.0:
        print("FAIL: NaN/divergence detected")
        raise RuntimeError("FAIL: NaN/divergence detected")

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(hyperparameters["gradient_clip_norm"]),
    )
    grad_norm_value = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
    if not np.isfinite(grad_norm_value):
        print("FAIL: NaN/divergence detected")
        raise RuntimeError("FAIL: NaN/divergence detected")

    learning_rate = float(hyperparameters["learning_rate"])
    weight_decay = float(hyperparameters["weight_decay"])
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if weight_decay > 0.0:
                parameter.add_(parameter, alpha=-learning_rate * weight_decay)
            parameter.add_(parameter.grad, alpha=-learning_rate)
    return loss_value, grad_norm_value


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
            logits = model(images.to(device))
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

    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    history = []
    epochs_completed = 0

    for epoch in range(int(hyperparameters["num_epochs"])):
        if harness.should_stop():
            break

        model.train()
        epoch_losses = []
        epoch_grad_norms = []

        for images, labels in train_loader:
            if harness.should_stop():
                break
            batch_loss, batch_grad_norm = manual_sgd_step(
                model=model,
                images=images.to(device),
                labels=labels.to(device),
                hyperparameters=hyperparameters,
            )
            epoch_losses.append(batch_loss)
            epoch_grad_norms.append(batch_grad_norm)

        val_accuracy = evaluate_digit_accuracy(model, val_loader, device)
        epochs_completed += 1
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else float("nan")
        print(
            f"train seed={seed} epoch={epoch} "
            f"train_loss: {train_loss:.4f} grad_norm: {grad_norm:.4f} "
            f"val_digit_accuracy: {val_accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "grad_norm": grad_norm,
                "val_digit_accuracy": val_accuracy,
            }
        )
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

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
    corrupted = image.clone()
    noise_std = None
    occlusion_size = None
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
        corrupted + torch.from_numpy(rng.normal(0.0, noise_std, size=corrupted.shape)).float(),
        0.0,
        1.0,
    )

    y0 = None
    x0 = None
    factor = None
    if occlusion_size > 0:
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
        pointer = pointers[label]
        if pointer >= len(bucket):
            rng.shuffle(bucket)
            pointer = 0
        selected_index = bucket[pointer]
        pointers[label] = pointer + 1
        return test_images[selected_index]

    examples: list[TaskExample] = []

    local_sequence_length = int(hyperparameters["local_sequence_length"])
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

    nonlocal_sequence_length = int(hyperparameters["nonlocal_sequence_length"])
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
        for start in range(0, len(flat_images), int(batch_size)):
            batch = flat_images[start : start + int(batch_size)].to(device)
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