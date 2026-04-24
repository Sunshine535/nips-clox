"""Candidate feature extraction for CLOX-PCS calibrated selector.

GPT-5.5 Pro Task 7: compute fixed-size feature vector per candidate.

Features (all computed from the candidate pool, no labels):
- strategy one-hot over known strategy names
- own_confidence
- cluster_size                 — number of candidates in same answer cluster
- cluster_fraction             — cluster_size / total_candidates
- cluster_mean_confidence      — average confidence of candidates in cluster
- cluster_strategy_diversity   — number of distinct strategies supporting cluster
- candidate_tokens             — normalized (log-scaled)
- cluster_tokens_share         — cluster total tokens / pool total tokens
- has_logprob                  — 1 if logprob_sum available
- logprob_sum_norm             — logprob_sum / tokens (else 0)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

KNOWN_STRATEGIES = [
    "standard_cot",
    "sc_sample",
    "self_consistency",
    "compute_matched_sc",
    "targeted_repair",
    "random_repair",
    "backward_cloze",
    "full_regeneration",
    "hierarchical_repair",
    "clox_adaptive",
    "bav",
]

FEATURE_NAMES = (
    [f"is_{s}" for s in KNOWN_STRATEGIES]
    + [
        "own_confidence",
        "cluster_size",
        "cluster_fraction",
        "cluster_mean_confidence",
        "cluster_strategy_diversity",
        "candidate_tokens_log",
        "cluster_tokens_share",
        "has_logprob",
        "logprob_sum_norm",
    ]
)


def _cluster_stats(cands: list[dict]) -> dict[int, dict[str, float]]:
    """Pre-compute cluster aggregates."""
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for c in cands:
        by_cluster[c["answer_cluster_id"]].append(c)
    total_tokens = sum(c.get("tokens", 0) for c in cands) or 1
    stats = {}
    for cid, members in by_cluster.items():
        stats[cid] = {
            "size": len(members),
            "fraction": len(members) / max(len(cands), 1),
            "mean_confidence": sum(c.get("confidence", 0.0) for c in members) / len(members),
            "strategy_diversity": len({c["strategy"] for c in members}),
            "tokens_share": sum(c.get("tokens", 0) for c in members) / total_tokens,
        }
    return stats


def extract_features(candidate: dict, pool: list[dict]) -> list[float]:
    """Return a fixed-size feature vector aligned with FEATURE_NAMES."""
    cstats = _cluster_stats(pool)
    cid = candidate["answer_cluster_id"]
    stat = cstats.get(cid, {"size": 1, "fraction": 1.0,
                           "mean_confidence": 0.0,
                           "strategy_diversity": 1,
                           "tokens_share": 1.0})

    one_hot = [1.0 if candidate["strategy"] == s else 0.0 for s in KNOWN_STRATEGIES]
    tokens = max(candidate.get("tokens", 0), 0)
    tokens_log = math.log1p(tokens)
    logprob_sum = candidate.get("logprob_sum")
    has_lp = 1.0 if logprob_sum is not None else 0.0
    lp_norm = (logprob_sum / tokens) if (logprob_sum is not None and tokens > 0) else 0.0

    return one_hot + [
        float(candidate.get("confidence", 0.0)),
        float(stat["size"]),
        float(stat["fraction"]),
        float(stat["mean_confidence"]),
        float(stat["strategy_diversity"]),
        float(tokens_log),
        float(stat["tokens_share"]),
        float(has_lp),
        float(lp_norm),
    ]


def build_feature_matrix(rows: list[dict]) -> tuple[list[list[float]], list[str]]:
    """Rows are per-example records with key 'candidates': list[dict].

    Returns (X, feature_names) where each X[i] is one candidate's features
    alongside a mirrored 'correct' label array (see build_labels).
    """
    X = []
    for row in rows:
        pool = row["candidates"]
        for c in pool:
            X.append(extract_features(c, pool))
    return X, list(FEATURE_NAMES)


def build_labels(rows: list[dict], ground_truth_key: str = "ground_truth",
                 answer_type_key: str = "answer_type") -> list[int]:
    """Label each candidate 1 if its normalized_answer matches ground_truth,
    using answer_extraction.check_answer_strict. MUST be called on calibration
    split only — test labels must never reach the selector."""
    from answer_extraction import check_answer_strict
    y = []
    for row in rows:
        gt = row[ground_truth_key]
        atype = row.get(answer_type_key, "text")
        for c in row["candidates"]:
            pred = c.get("normalized_answer", "")
            y.append(1 if check_answer_strict(str(pred), str(gt), atype) else 0)
    return y
