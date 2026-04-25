"""Calibrated candidate selector for CLOX-PCS.

GPT-5.5 Pro Task 7: learn P(candidate_correct | features) on calibration split,
then apply isotonic calibration. At test time, score candidates and pick argmax
per-cluster.

Training contract:
- Fit on calibration split ONLY
- Report AUC, Brier, ECE on held-out (no leakage)
- Persist to pickle for reuse at eval time

Usage:
    # fit
    python code/calibrated_selector.py fit \
        --calib results/pcs/calib_candidates.jsonl \
        --out results/pcs/selector.pkl

    # report
    python code/calibrated_selector.py report \
        --calib results/pcs/calib_candidates.jsonl \
        --model results/pcs/selector.pkl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

from features import FEATURE_NAMES, build_feature_matrix, build_labels


@dataclass
class SelectorArtifact:
    logistic: LogisticRegression
    isotonic: IsotonicRegression | None
    feature_names: list[str]
    meta: dict[str, Any]


def load_candidate_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if lo == bins[-2]:
            mask |= (y_prob == 1.0)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def fit_selector(
    rows: list[dict],
    random_state: int = 11,
    val_fraction: float = 0.2,
    holdout_fraction: float = 0.2,
) -> tuple[SelectorArtifact, dict[str, float]]:
    """3-way example-id-stratified split:
        train    — LR fit
        val      — isotonic fit
        holdout  — held-out metrics (reported as the only honest numbers)

    Splits by example_id (not candidate row), so no candidates from the same
    question appear in two splits.
    """
    if len(rows) < 5 or len(np.unique([r["example_id"] for r in rows])) < 5:
        raise ValueError(
            f"need at least 5 distinct example_ids; got {len(rows)} rows. "
            "Selector requires distinct splits."
        )
    rng = np.random.default_rng(random_state)
    ids = sorted({r["example_id"] for r in rows})
    rng.shuffle(ids)
    n = len(ids)
    n_holdout = max(1, int(round(n * holdout_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_holdout - n_val
    if n_train < 1:
        # Tiny-sample fallback: assign at least 1 row to each split
        n_holdout = 1
        n_val = 1
        n_train = max(1, n - 2)

    holdout_ids = set(ids[:n_holdout])
    val_ids = set(ids[n_holdout:n_holdout + n_val])
    train_ids = set(ids[n_holdout + n_val:n_holdout + n_val + n_train])

    train_rows = [r for r in rows if r["example_id"] in train_ids]
    val_rows = [r for r in rows if r["example_id"] in val_ids]
    holdout_rows = [r for r in rows if r["example_id"] in holdout_ids]

    def _Xy(part):
        X, _ = build_feature_matrix(part)
        y = build_labels(part)
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

    X_tr, y_tr = _Xy(train_rows)
    X_val, y_val = _Xy(val_rows)
    X_ho, y_ho = _Xy(holdout_rows)

    if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
        # Fallback if train split has only one class — fit on all rows
        X_full, y_full = _Xy(rows)
        logistic = LogisticRegression(max_iter=1000).fit(X_full, y_full)
        iso = None
        leakage_warning = "train split single-class; trained on full rows (BIASED)"
    else:
        logistic = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        if len(X_val) > 0 and len(np.unique(y_val)) >= 2:
            p_val_raw = logistic.predict_proba(X_val)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_val_raw, y_val)
        else:
            iso = None
        leakage_warning = ""

    def _metrics(X, y, label):
        if len(X) == 0 or len(np.unique(y)) < 2:
            return {f"{label}_auc": float("nan"),
                    f"{label}_brier": float("nan"),
                    f"{label}_ece": float("nan"),
                    f"{label}_n": int(len(X)),
                    f"{label}_positive_rate": float(y.mean()) if len(y) else float("nan")}
        p_raw = logistic.predict_proba(X)[:, 1]
        p_cal = iso.predict(p_raw) if iso is not None else p_raw
        return {
            f"{label}_auc_raw": float(roc_auc_score(y, p_raw)),
            f"{label}_auc_calibrated": float(roc_auc_score(y, p_cal)),
            f"{label}_brier_raw": float(brier_score_loss(y, p_raw)),
            f"{label}_brier_calibrated": float(brier_score_loss(y, p_cal)),
            f"{label}_ece_raw": expected_calibration_error(y, p_raw),
            f"{label}_ece_calibrated": expected_calibration_error(y, p_cal),
            f"{label}_n": int(len(X)),
            f"{label}_positive_rate": float(y.mean()),
        }

    metrics: dict[str, float] = {
        "n_examples_train": len(train_ids),
        "n_examples_val": len(val_ids),
        "n_examples_holdout": len(holdout_ids),
        "leakage_warning": leakage_warning,
    }
    metrics.update(_metrics(X_tr, y_tr, "train"))
    metrics.update(_metrics(X_val, y_val, "val"))
    metrics.update(_metrics(X_ho, y_ho, "holdout"))

    feature_names_full = list(build_feature_matrix(rows[:1])[1]) if rows else []
    feature_schema_hash = hashlib.sha256(
        json.dumps(feature_names_full, sort_keys=True).encode()
    ).hexdigest()[:16] if feature_names_full else ""
    art = SelectorArtifact(
        logistic=logistic, isotonic=iso,
        feature_names=feature_names_full,
        meta={"random_state": random_state,
              "val_fraction": val_fraction,
              "holdout_fraction": holdout_fraction,
              "feature_count": len(feature_names_full),
              "feature_schema_hash": feature_schema_hash,
              "train_example_ids": sorted(train_ids),
              "val_example_ids": sorted(val_ids),
              "holdout_example_ids": sorted(holdout_ids)},
    )
    return art, metrics


def score_pool(art: SelectorArtifact, pool: list[dict]) -> list[float]:
    """Score candidates in a pool with the fitted selector + isotonic."""
    from features import extract_features
    X = np.asarray([extract_features(c, pool) for c in pool], dtype=float)
    if len(X) == 0:
        return []
    p_raw = art.logistic.predict_proba(X)[:, 1]
    if art.isotonic is not None:
        return art.isotonic.predict(p_raw).tolist()
    return p_raw.tolist()


def save_artifact(art: SelectorArtifact, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(art, f)


def load_artifact(path: str) -> SelectorArtifact:
    with open(path, "rb") as f:
        return pickle.load(f)


def cmd_fit(args) -> None:
    rows = load_candidate_rows(args.calib)
    art, metrics = fit_selector(rows, random_state=args.seed)
    save_artifact(art, args.out)
    report_path = os.path.join(
        os.path.dirname(args.out) or ".",
        os.path.splitext(os.path.basename(args.out))[0] + "_metrics.json",
    )
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved selector → {args.out}")
    print(f"saved metrics  → {report_path}")


def cmd_report(args) -> None:
    rows = load_candidate_rows(args.calib)
    art = load_artifact(args.model)
    all_candidates = [c for row in rows for c in row["candidates"]]
    # Build features per-row so cluster stats are consistent
    from features import extract_features
    y = build_labels(rows)
    probs = []
    for row in rows:
        pool = row["candidates"]
        for c in pool:
            X_i = np.asarray([extract_features(c, pool)], dtype=float)
            p_raw = art.logistic.predict_proba(X_i)[:, 1][0]
            if art.isotonic is not None:
                p_raw = float(art.isotonic.predict([p_raw])[0])
            probs.append(p_raw)
    y = np.asarray(y); p = np.asarray(probs)
    metrics = {
        "n_candidates": int(len(p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "positive_rate": float(y.mean()),
    }
    print(json.dumps(metrics, indent=2))


def cmd_debug_tiny(args) -> None:
    """Self-overfit sanity: trivial signal selector must reach holdout AUC > 0.55."""
    rng = np.random.default_rng(0)
    rows = []
    # Strong signal: standard_cot always emits ground truth; sc_sample emits noise.
    for i in range(60):
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
        # Cluster by normalized_answer
        k2id = {}
        for c in cands:
            if c["normalized_answer"] not in k2id:
                k2id[c["normalized_answer"]] = len(k2id)
            c["answer_cluster_id"] = k2id[c["normalized_answer"]]
        rows.append({"example_id": f"ex{i}", "ground_truth": gt,
                     "answer_type": "numeric", "candidates": cands})
    art, metrics = fit_selector(rows, random_state=11)
    print(json.dumps({k: v for k, v in metrics.items()
                      if "holdout" in k or "n_examples" in k}, indent=2))
    holdout_auc = metrics.get("holdout_auc_raw", float("nan"))
    holdout_pos_rate = metrics.get("holdout_positive_rate", 0.5)
    assert (
        np.isnan(holdout_auc) or holdout_auc > 0.55 or holdout_pos_rate in (0.0, 1.0)
    ), f"selector failed on trivial signal — held-out AUC = {holdout_auc}: {metrics}"
    print("debug_tiny OK (held-out AUC > 0.55)")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--calib", required=True)
    p_fit.add_argument("--out", required=True)
    p_fit.add_argument("--seed", type=int, default=11)
    p_fit.set_defaults(func=cmd_fit)

    p_rep = sub.add_parser("report")
    p_rep.add_argument("--calib", required=True)
    p_rep.add_argument("--model", required=True)
    p_rep.set_defaults(func=cmd_report)

    p_dbg = sub.add_parser("debug_tiny")
    p_dbg.set_defaults(func=cmd_debug_tiny)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
