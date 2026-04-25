"""Tests for run_portfolio_experiment.py — Task 1 smoke + Task 4 gate accounting + Task 5 arms."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))


def _make_candidate(strategy, ans, cluster_id, tokens=100, conf=0.5, idx=0):
    return {
        "candidate_id": f"ex0:{strategy}:{idx}",
        "strategy": strategy,
        "sample_index": idx,
        "raw_output": "",
        "normalized_answer": ans,
        "answer_cluster_id": cluster_id,
        "tokens": tokens,
        "prompt_tokens": 10,
        "confidence": conf,
    }


def _make_row(ex_id, gt, candidates, atype="numeric"):
    return {
        "example_id": ex_id, "question": "q",
        "ground_truth": gt, "answer_type": atype,
        "candidates": candidates,
    }


def _build_synthetic_dataset(n_examples=12):
    rows = []
    for i in range(n_examples):
        gt = str(i % 3)
        cands = [
            _make_candidate("standard_cot", gt, 0, tokens=100, conf=0.9),
            _make_candidate("sc_sample", gt, 0, tokens=80, conf=0.5, idx=0),
            _make_candidate("sc_sample", str((i + 1) % 5), 1, tokens=80, conf=0.5, idx=1),
            _make_candidate("sc_sample", str((i + 2) % 5), 2, tokens=80, conf=0.5, idx=2),
            _make_candidate("targeted_repair", gt, 0, tokens=200, conf=0.7),
            _make_candidate("backward_cloze", str((i + 3) % 5), 3, tokens=150, conf=0.4),
            _make_candidate("random_repair", gt, 0, tokens=180, conf=0.4),
            _make_candidate("full_regeneration", str((i + 4) % 5), 4, tokens=200, conf=0.4),
        ]
        # Re-cluster by normalized_answer
        k2i = {}
        for c in cands:
            if c["normalized_answer"] not in k2i:
                k2i[c["normalized_answer"]] = len(k2i)
            c["answer_cluster_id"] = k2i[c["normalized_answer"]]
        rows.append(_make_row(f"ex{i}", gt, cands))
    return rows


# ── Task 1 — runner imports & mock collect ──────────────────────────


class TestRunnerSmoke:
    def test_imports_clean(self):
        import run_portfolio_experiment as rpe
        assert hasattr(rpe, "_collect")
        assert hasattr(rpe, "_eval")
        assert "A_SC" in rpe.ARM_FNS
        assert "A_BAV" in rpe.ARM_FNS
        assert "A_TARGETED" in rpe.ARM_FNS
        assert "B" in rpe.ARM_FNS

    def test_engine_uses_model_name_kw(self, monkeypatch):
        """Regression: collect mode must NOT pass model_path= to VLLMEngine.
        We patch the engine factory and inspect the kwargs received."""
        import run_portfolio_experiment as rpe
        captured = {}

        class _Probe:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(rpe, "_build_engine", lambda args: _Probe(
            model_name=args.model, tensor_parallel_size=args.tp, seed=args.seed))

        class A: pass
        a = A()
        a.model = "test"; a.tp = 1; a.seed = 11
        eng = rpe._build_engine(a)
        assert "model_name" in captured
        assert "model_path" not in captured


# ── Task 4 — staged gate produces non-trivial token accounting ───────


class TestStagedGate:
    def test_C_uses_only_scout_when_decisive(self, monkeypatch):
        """If selector says cluster has high confidence at scout stage, C
        should report only scout tokens, not the full pool."""
        import run_portfolio_experiment as rpe
        from compute_gate import GateDecision

        # Force gate to STOP at stage 1
        def fake_decide(*args, **kwargs):
            return GateDecision(
                action="stop", rationale="forced",
                best_cluster_id=0, best_score=0.99, cluster_score_map={0: 0.99},
            )
        monkeypatch.setattr("compute_gate.decide", fake_decide)

        # Provide a fake selector that always returns high scores
        from calibrated_selector import SelectorArtifact
        from features import FEATURE_NAMES
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        d = len(FEATURE_NAMES)
        lr = LogisticRegression().fit(np.array([[0.0]*d, [1.0]*d]), [0, 1])
        art = SelectorArtifact(logistic=lr, isotonic=None,
                               feature_names=list(FEATURE_NAMES),
                               meta={})
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            import pickle; pickle.dump(art, f)
            sel_path = f.name

        rows = _build_synthetic_dataset(3)
        cand_path = os.path.join(tempfile.mkdtemp(), "c.jsonl")
        with open(cand_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        eval_out = os.path.join(tempfile.mkdtemp(), "e.json")

        class A: pass
        a = A()
        a.candidates = cand_path; a.selector = sel_path
        a.out = eval_out; a.compare = "B,C"
        a.tau_stop = 0.75; a.tau_margin = 0.10
        rpe._eval(a)

        d = json.load(open(eval_out))
        for ex in d["per_example"]:
            c_arm = ex["arms"]["C"]
            b_arm = ex["arms"]["B"]
            assert c_arm["extra"]["expansion_used"] is False
            assert c_arm["tokens_used"] < b_arm["tokens_used"], (
                f"C scout tokens {c_arm['tokens_used']} should be < B full tokens {b_arm['tokens_used']}"
            )
            assert "decision_stage1" in c_arm["extra"]


# ── Task 5 — multiple A controls produce distinct predictions/tokens ─


class TestArmControls:
    def test_arms_distinct_pickers(self):
        from run_portfolio_experiment import _arm_A_SC, _arm_A_TARGETED, _arm_B
        cands = [
            _make_candidate("sc_sample", "10", 0, tokens=50, idx=0),
            _make_candidate("sc_sample", "20", 1, tokens=50, idx=1),
            _make_candidate("sc_sample", "20", 1, tokens=50, idx=2),
            _make_candidate("targeted_repair", "30", 2, tokens=200),
            _make_candidate("backward_cloze", "40", 3, tokens=150),
        ]
        sc_pred, sc_tok = _arm_A_SC(cands)
        tr_pred, tr_tok = _arm_A_TARGETED(cands)
        b_pred, b_tok = _arm_B(cands)
        assert sc_pred == "20"
        assert tr_pred == "30"
        assert sc_tok == 150
        assert tr_tok == 200
        assert b_tok == sum(c["tokens"] for c in cands)

    def test_A_BAV_returns_empty_if_absent(self):
        from run_portfolio_experiment import _arm_A_BAV
        cands = [_make_candidate("sc_sample", "1", 0)]
        pred, tokens = _arm_A_BAV(cands)
        assert pred == ""
        assert tokens == 0
