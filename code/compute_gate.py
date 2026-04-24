"""Value-of-compute gate for CLOX-PCS.

GPT-5.5 Pro Task 8: decide stop / continue based on current best cluster
confidence. Conservative v1: stop if cluster calibrated score >= tau_stop
OR remaining budget does not allow another strategy call.

Inputs:
    candidates: list of dicts, each with cluster_id, calibrated_score, tokens
    remaining_budget: int (tokens)
    next_strategy_cost: int (est tokens)

Outputs:
    GateDecision(action, rationale, best_cluster_id, best_score)

This is a decision function, not a model — it will be logged per-example so
we can compute precision/recall of stops post-hoc.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal


Action = Literal["stop", "continue"]


@dataclass
class GateDecision:
    action: Action
    rationale: str
    best_cluster_id: int
    best_score: float
    cluster_score_map: dict[int, float]


def aggregate_cluster_scores(
    candidates: list[dict],
) -> dict[int, float]:
    """Cluster score = max calibrated candidate score within cluster.

    Rationale: a cluster is as good as its best-calibrated member. Using max
    (not mean) avoids penalizing clusters that collect poor-support siblings
    while still containing one high-confidence candidate.
    """
    by_cluster: dict[int, list[float]] = defaultdict(list)
    for c in candidates:
        by_cluster[c["answer_cluster_id"]].append(
            float(c.get("calibrated_score", c.get("confidence", 0.0)))
        )
    return {cid: (max(scores) if scores else 0.0) for cid, scores in by_cluster.items()}


def decide(
    candidates: list[dict],
    remaining_budget: int,
    next_strategy_cost: int,
    tau_stop: float = 0.75,
    tau_margin: float = 0.10,
) -> GateDecision:
    """Conservative value-of-compute gate.

    Stop if:
      - best cluster score >= tau_stop, AND
      - best cluster score exceeds runner-up by tau_margin (decisive)
    OR remaining budget < next_strategy_cost.

    Otherwise continue.
    """
    cmap = aggregate_cluster_scores(candidates)
    if not cmap:
        return GateDecision(
            action="stop", rationale="no candidates",
            best_cluster_id=-1, best_score=0.0, cluster_score_map={},
        )

    ordered = sorted(cmap.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = best_score - runner_up

    # Budget constraint overrides
    if remaining_budget < next_strategy_cost:
        return GateDecision(
            action="stop",
            rationale=f"budget exhausted (remaining={remaining_budget} < next={next_strategy_cost})",
            best_cluster_id=best_id, best_score=best_score, cluster_score_map=cmap,
        )

    if best_score >= tau_stop and margin >= tau_margin:
        return GateDecision(
            action="stop",
            rationale=f"confident: score={best_score:.3f} >= {tau_stop}, margin={margin:.3f}",
            best_cluster_id=best_id, best_score=best_score, cluster_score_map=cmap,
        )

    return GateDecision(
        action="continue",
        rationale=f"uncertain: best={best_score:.3f}, margin={margin:.3f}, tau_stop={tau_stop}",
        best_cluster_id=best_id, best_score=best_score, cluster_score_map=cmap,
    )


def pick_best_answer(candidates: list[dict]) -> dict:
    """Return the candidate from the highest-scoring cluster (argmax within cluster)."""
    if not candidates:
        return {}
    cmap = aggregate_cluster_scores(candidates)
    best_cid = max(cmap, key=cmap.get)
    in_cluster = [c for c in candidates if c["answer_cluster_id"] == best_cid]
    in_cluster.sort(key=lambda c: float(c.get("calibrated_score", c.get("confidence", 0.0))),
                    reverse=True)
    return in_cluster[0]
