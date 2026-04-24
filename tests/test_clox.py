"""Tests for CLOX legacy v1 (strategy budget matching, BBH macro-averaging, topology).

Archived per GPT-5.5 Pro Task 1 — targets `archive/legacy_clox_v1/strategies.py`.
Skipped in the PCS test suite; kept for historical audit only.
"""
from __future__ import annotations

import sys
import os

import pytest

pytestmark = pytest.mark.skip(reason="Legacy CLOX v1 — archived per GPT-5.5 Pro Task 1.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "archive", "legacy_clox_v1"))

import numpy as np

from benchmarks import compute_bbh_macro_average
from evaluation import compute_task_topology_metrics
from strategies import (
    InferenceStrategy,
    SelfConsistency,
    AnswerAnchoredBackwardCloze,
    FullRationaleRegeneration,
    UncertaintyTargetedMaskedRepair,
    RandomSpanMaskedRepair,
    HierarchicalMaskedRepair,
    StandardCoT,
)


# ── Helpers ────────────────────────────────────────────────────────────


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, **kw):
        return text.split()

    def __call__(self, text, **kw):
        ids = list(range(len(text.split())))
        import torch

        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def decode(self, ids, **kw):
        return "Step 1: compute. Step 2: add. The answer is 42."


class _FakeModel:
    device = "cpu"

    def eval(self):
        pass

    def generate(self, input_ids, attention_mask=None, **kw):
        import torch

        max_new = kw.get("max_new_tokens", 10)
        seq = torch.cat(
            [input_ids, torch.zeros(1, max_new, dtype=torch.long)], dim=1
        )

        class _Out:
            sequences = seq
            scores = tuple(
                torch.randn(1, 100) for _ in range(max_new)
            )

        return _Out()


# ── 1. Strategy budget matching ───────────────────────────────────────


class TestStrategyBudgetMatching:
    """token_budget should propagate and constrain all strategies."""

    def test_self_consistency_derives_k_from_budget(self):
        sc = SelfConsistency(k_samples=5, temperature=0.7)
        result = sc.run(
            _FakeModel(),
            _FakeTokenizer(),
            "What is 2+2?",
            max_new_tokens=100,
            token_budget=300,
        )
        assert result.total_tokens > 0
        assert result.strategy_name == "self_consistency"

    def test_self_consistency_k_at_least_1(self):
        sc = SelfConsistency(k_samples=10, temperature=0.7)
        result = sc.run(
            _FakeModel(),
            _FakeTokenizer(),
            "What is 2+2?",
            max_new_tokens=500,
            token_budget=50,
        )
        assert result.total_tokens > 0

    def test_backward_cloze_derives_candidates_from_budget(self):
        bc = AnswerAnchoredBackwardCloze(n_candidates=5)
        result = bc.run(
            _FakeModel(),
            _FakeTokenizer(),
            "What is 2+2?",
            max_new_tokens=100,
            token_budget=300,
        )
        assert result.total_tokens > 0
        assert result.strategy_name == "backward_cloze"

    def test_standard_cot_respects_budget(self):
        cot = StandardCoT(few_shot=False)
        result = cot.run(
            _FakeModel(),
            _FakeTokenizer(),
            "What is 2+2?",
            max_new_tokens=512,
            token_budget=200,
        )
        assert result.completion_tokens <= 200
        assert result.total_tokens > 0

    def test_full_regeneration_respects_budget(self):
        fr = FullRationaleRegeneration()
        result = fr.run(
            _FakeModel(),
            _FakeTokenizer(),
            "What is 2+2?",
            max_new_tokens=512,
            token_budget=200,
        )
        assert result.total_tokens > 0

    def test_no_budget_uses_default(self):
        cot = StandardCoT(few_shot=False)
        result = cot.run(
            _FakeModel(),
            _FakeTokenizer(),
            "What is 2+2?",
            max_new_tokens=256,
        )
        assert result.completion_tokens <= 256

    def test_all_strategies_accept_token_budget_kwarg(self):
        """Every concrete strategy must accept token_budget without error."""
        strats = [
            StandardCoT(few_shot=False),
            SelfConsistency(k_samples=2),
            AnswerAnchoredBackwardCloze(n_candidates=2),
            UncertaintyTargetedMaskedRepair(),
            RandomSpanMaskedRepair(),
            FullRationaleRegeneration(),
            HierarchicalMaskedRepair(),
        ]
        for s in strats:
            result = s.run(
                _FakeModel(),
                _FakeTokenizer(),
                "What is 1+1?",
                max_new_tokens=100,
                token_budget=200,
            )
            assert result.total_tokens > 0, f"{s.name} returned 0 tokens"


# ── 2. BBH subtask macro-averaging ────────────────────────────────────


class TestBBHMacroAverage:
    def test_macro_average_equal_subtasks(self):
        results = [
            {"benchmark": "bbh_navigate", "correct": True, "metadata": {"subtask": "navigate"}},
            {"benchmark": "bbh_navigate", "correct": False, "metadata": {"subtask": "navigate"}},
            {"benchmark": "bbh_date_understanding", "correct": True, "metadata": {"subtask": "date_understanding"}},
            {"benchmark": "bbh_date_understanding", "correct": True, "metadata": {"subtask": "date_understanding"}},
        ]
        out = compute_bbh_macro_average(results)
        assert out["n_subtasks"] == 2
        assert out["per_subtask"]["navigate"] == pytest.approx(0.5)
        assert out["per_subtask"]["date_understanding"] == pytest.approx(1.0)
        assert out["macro_average"] == pytest.approx(0.75)

    def test_macro_average_imbalanced(self):
        results = (
            [{"benchmark": "bbh_a", "correct": True, "metadata": {"subtask": "a"}}] * 10
            + [{"benchmark": "bbh_b", "correct": False, "metadata": {"subtask": "b"}}] * 2
        )
        out = compute_bbh_macro_average(results)
        assert out["per_subtask"]["a"] == pytest.approx(1.0)
        assert out["per_subtask"]["b"] == pytest.approx(0.0)
        assert out["macro_average"] == pytest.approx(0.5)

    def test_macro_average_empty(self):
        out = compute_bbh_macro_average([])
        assert out["macro_average"] == 0.0
        assert out["n_subtasks"] == 0

    def test_fallback_to_benchmark_prefix(self):
        results = [
            {"benchmark": "bbh_navigate", "correct": True, "metadata": {}},
            {"benchmark": "bbh_navigate", "correct": True, "metadata": {}},
        ]
        out = compute_bbh_macro_average(results)
        assert out["n_subtasks"] == 1
        assert out["per_subtask"]["navigate"] == pytest.approx(1.0)

    def test_single_subtask(self):
        results = [
            {"benchmark": "bbh_x", "correct": True, "metadata": {"subtask": "x"}},
            {"benchmark": "bbh_x", "correct": False, "metadata": {"subtask": "x"}},
            {"benchmark": "bbh_x", "correct": True, "metadata": {"subtask": "x"}},
        ]
        out = compute_bbh_macro_average(results)
        assert out["macro_average"] == pytest.approx(2.0 / 3.0)


# ── 3. Topology metric computation ────────────────────────────────────


class TestTopologyMetrics:
    def test_basic_topology(self):
        per_example = {
            "standard_cot": [
                {
                    "correct": True,
                    "step_metadata": [{"step": 0}, {"step": 1}],
                    "logprobs": [-0.5, -1.0, -0.3],
                },
                {
                    "correct": False,
                    "step_metadata": [{"step": 0}],
                    "logprobs": [-2.0, -0.1, -3.0],
                },
            ],
            "self_consistency": [
                {
                    "correct": True,
                    "step_metadata": [{"step": 0}, {"step": 1}, {"step": 2}],
                    "logprobs": [-0.2, -0.3, -0.1],
                },
                {
                    "correct": True,
                    "step_metadata": [{"step": 0}],
                    "logprobs": [-0.5, -0.5],
                },
            ],
        }
        out = compute_task_topology_metrics(per_example)
        assert "estimated_epl" in out
        assert "estimated_recoverability" in out
        assert "mean_steps" in out
        assert out["n_examples"] == 4
        assert out["mean_steps"] > 0

    def test_recoverability_with_agreement(self):
        same = [
            {"correct": True, "step_metadata": [{"s": 0}], "logprobs": [-0.1]},
            {"correct": True, "step_metadata": [{"s": 0}], "logprobs": [-0.1]},
        ]
        per_example = {"a": same, "b": same}
        out = compute_task_topology_metrics(per_example)
        assert out["estimated_recoverability"] == pytest.approx(1.0)

    def test_recoverability_with_disagreement(self):
        per_example = {
            "a": [
                {"correct": True, "step_metadata": [{"s": 0}], "logprobs": [-0.1]},
                {"correct": False, "step_metadata": [{"s": 0}], "logprobs": [-0.1]},
            ],
            "b": [
                {"correct": False, "step_metadata": [{"s": 0}], "logprobs": [-0.1]},
                {"correct": True, "step_metadata": [{"s": 0}], "logprobs": [-0.1]},
            ],
        }
        out = compute_task_topology_metrics(per_example)
        assert out["estimated_recoverability"] < 1.0

    def test_epl_from_logprobs(self):
        per_example = {
            "cot": [
                {
                    "correct": False,
                    "step_metadata": [],
                    "logprobs": [-0.1, -0.2, -5.0, -0.3, -0.4],
                },
            ],
        }
        out = compute_task_topology_metrics(per_example)
        assert out["estimated_epl"] > 0

    def test_empty_input(self):
        out = compute_task_topology_metrics({})
        assert out["n_examples"] == 0
        assert out["mean_steps"] == pytest.approx(5.0)

    def test_single_strategy(self):
        per_example = {
            "only": [
                {"correct": True, "step_metadata": [{"s": 0}, {"s": 1}], "logprobs": [-0.5]},
            ],
        }
        out = compute_task_topology_metrics(per_example)
        assert out["estimated_recoverability"] == pytest.approx(0.5)
        assert out["n_examples"] == 1
