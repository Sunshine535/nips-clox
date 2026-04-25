"""Selector no-leakage tests — GPT-5.5 Pro Task 6."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from calibrated_selector import fit_selector


def _make_dataset(n=60):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        gt = str(i % 3)
        cands = [{
            "candidate_id": f"ex{i}:c{k}",
            "strategy": "standard_cot" if k == 0 else "random_repair",
            "sample_index": k,
            "raw_output": "",
            "normalized_answer": gt if k == 0 else str(rng.integers(0, 5)),
            "answer_cluster_id": -1,
            "tokens": 100, "prompt_tokens": 10,
            "confidence": 0.9 if k == 0 else 0.3,
        } for k in range(4)]
        k2id = {}
        for c in cands:
            if c["normalized_answer"] not in k2id:
                k2id[c["normalized_answer"]] = len(k2id)
            c["answer_cluster_id"] = k2id[c["normalized_answer"]]
        rows.append({"example_id": f"ex{i}", "ground_truth": gt,
                     "answer_type": "numeric", "candidates": cands})
    return rows


def test_no_example_id_leakage():
    """Same example_id must not appear in two different splits."""
    rows = _make_dataset(60)
    art, metrics = fit_selector(rows, random_state=11)
    train_ids = set(art.meta["train_example_ids"])
    val_ids = set(art.meta["val_example_ids"])
    holdout_ids = set(art.meta["holdout_example_ids"])
    assert not (train_ids & val_ids), "example_id leak: train ∩ val non-empty"
    assert not (train_ids & holdout_ids), "example_id leak: train ∩ holdout non-empty"
    assert not (val_ids & holdout_ids), "example_id leak: val ∩ holdout non-empty"


def test_three_splits_cover_all_examples():
    rows = _make_dataset(60)
    art, _ = fit_selector(rows, random_state=11)
    union = (set(art.meta["train_example_ids"])
             | set(art.meta["val_example_ids"])
             | set(art.meta["holdout_example_ids"]))
    assert len(union) == len({r["example_id"] for r in rows})


def test_metrics_include_holdout():
    rows = _make_dataset(60)
    _, metrics = fit_selector(rows, random_state=11)
    assert "holdout_auc_raw" in metrics
    assert "holdout_brier_raw" in metrics
    assert "holdout_ece_raw" in metrics
    assert metrics["holdout_n"] > 0


def test_feature_schema_hash_persisted():
    rows = _make_dataset(60)
    art, _ = fit_selector(rows, random_state=11)
    h = art.meta.get("feature_schema_hash", "")
    assert h, "feature_schema_hash must be set"
    assert len(h) == 16
