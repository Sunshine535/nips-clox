from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch

try:
    from experiment_harness import ExperimentHarness as NativeExperimentHarness
except Exception:
    NativeExperimentHarness = None


class LocalExperimentHarness:
    def __init__(self, time_budget: float, stop_fraction: float = 0.8) -> None:
        self.time_budget = float(time_budget)
        self.stop_fraction = float(stop_fraction)
        self.start_time = time.perf_counter()
        self.reported_metrics: dict[str, float] = {}

    def should_stop(self) -> bool:
        elapsed = time.perf_counter() - self.start_time
        return elapsed >= self.time_budget * self.stop_fraction

    def check_value(self, value: float, name: str) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except Exception:
            return False

    def report_metric(self, name: str, value: float) -> None:
        if self.check_value(value, name):
            self.reported_metrics[str(name)] = float(value)

    def finalize(self) -> None:
        return None


def build_harness(time_budget: float, stop_fraction: float = 0.8):
    if NativeExperimentHarness is not None:
        return NativeExperimentHarness(time_budget=time_budget)
    return LocalExperimentHarness(time_budget=time_budget, stop_fraction=stop_fraction)


def prepare_runtime_environment(workdir: str) -> None:
    runtime_tmp = os.path.join(workdir, ".runtime_tmp")
    try:
        os.makedirs(runtime_tmp, exist_ok=True)
        for key in ("TMPDIR", "TMP", "TEMP", "TORCHINDUCTOR_CACHE_DIR"):
            os.environ.setdefault(key, runtime_tmp)
    except OSError:
        pass


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Alias per GPT-5.5 Pro Task 4 spec
set_global_seed = set_all_seeds


def stable_hash_seed(*parts: object) -> int:
    """SHA256-based stable seed. Stable across processes (unlike Python hash)."""
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def stable_int_seed(*parts: object) -> int:
    """Back-compat shim — delegates to stable_hash_seed."""
    return stable_hash_seed(*parts)


def logsumexp(values) -> float:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return float("-inf")
    max_value = float(np.max(values))
    if not np.isfinite(max_value):
        return max_value
    return float(max_value + np.log(np.exp(values - max_value).sum()))


def mean_std(values) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std())


def bootstrap_ci(values, num_samples: int = 200, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    single = None
    if values.size == 1:
        single = float(values[0])
        return single, single
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(num_samples)):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(sample.mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_ci(
    a_values,
    b_values,
    num_samples: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    a_values = np.asarray(list(a_values), dtype=float)
    b_values = np.asarray(list(b_values), dtype=float)
    count = min(a_values.size, b_values.size)
    if count == 0:
        return float("nan"), float("nan")
    diffs = a_values[:count] - b_values[:count]
    return bootstrap_ci(diffs, num_samples=num_samples, seed=seed)


def cohen_d(a_values, b_values) -> float:
    a_values = np.asarray(list(a_values), dtype=float)
    b_values = np.asarray(list(b_values), dtype=float)
    a_values = a_values[np.isfinite(a_values)]
    b_values = b_values[np.isfinite(b_values)]
    if a_values.size == 0 or b_values.size == 0:
        return float("nan")
    diffs = None
    denom = None
    if a_values.size == b_values.size and a_values.size > 1:
        diffs = a_values - b_values
        denom = float(diffs.std(ddof=1))
        if denom == 0.0:
            return 0.0
        return float(diffs.mean() / denom)
    pooled_num = ((a_values.size - 1) * a_values.var(ddof=1)) + ((b_values.size - 1) * b_values.var(ddof=1))
    pooled_den = max(a_values.size + b_values.size - 2, 1)
    pooled_std = float(math.sqrt(max(pooled_num / pooled_den, 0.0)))
    if pooled_std == 0.0:
        return 0.0
    return float((a_values.mean() - b_values.mean()) / pooled_std)


def to_serializable(value):
    if isinstance(value, dict):
        return {str(key): to_serializable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(inner) for inner in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)