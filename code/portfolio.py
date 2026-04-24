"""CLOX-PCS Portfolio Candidate Generator.

GPT-5.5 Pro Task 6: collect heterogeneous candidate answers from multiple strategies.

Each candidate records:
- candidate_id       — per-example unique
- strategy           — name of generator
- sample_index       — index within strategy output
- raw_output         — full reasoning text
- normalized_answer  — extracted + normalized (answer-type-aware)
- answer_cluster_id  — equivalence class among candidates
- tokens             — actual completion tokens
- prompt_tokens
- confidence         — strategy-reported confidence
- logprob_sum        — if available

Candidate pool is the unit PCS selects over. Strategies remain untouched.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

from answer_extraction import extract_answer_typed
from engine import VLLMEngine, extract_answer
from strategies_v2 import STRATEGY_REGISTRY, StrategyResult, build_strategy


@dataclass
class Candidate:
    candidate_id: str
    strategy: str
    sample_index: int
    raw_output: str
    normalized_answer: str
    answer_cluster_id: int
    tokens: int
    prompt_tokens: int
    confidence: float
    logprob_sum: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_for_cluster(answer: str) -> str:
    if answer is None:
        return ""
    return str(answer).strip().lower().replace(" ", "")


def cluster_candidates(cands: list[Candidate]) -> list[Candidate]:
    """Assign answer_cluster_id by normalized answer."""
    key_to_id: dict[str, int] = {}
    out = []
    for c in cands:
        key = normalize_for_cluster(c.normalized_answer)
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        c.answer_cluster_id = key_to_id[key]
        out.append(c)
    return out


def _to_candidate(
    example_id: str,
    strategy: str,
    idx: int,
    raw: str,
    tokens: int,
    prompt_tokens: int,
    confidence: float,
    answer_type: str,
    logprob_sum: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Candidate:
    normalized = extract_answer_typed(raw, answer_type)
    return Candidate(
        candidate_id=f"{example_id}:{strategy}:{idx}",
        strategy=strategy,
        sample_index=idx,
        raw_output=raw,
        normalized_answer=normalized,
        answer_cluster_id=-1,
        tokens=int(tokens),
        prompt_tokens=int(prompt_tokens),
        confidence=float(confidence),
        logprob_sum=logprob_sum,
        extra=extra or {},
    )


def run_portfolio(
    engine: VLLMEngine,
    example_id: str,
    question: str,
    answer_type: str = "numeric",
    strategies: list[str] | None = None,
    sc_k: int = 8,
    sc_temperature: float = 0.7,
    max_tokens: int = 512,
    few_shot: str = "",
) -> list[Candidate]:
    """Collect all candidates from the portfolio.

    Special case: SC is expanded to k per-sample candidates (one per sample), not
    one majority-vote candidate. For SC we call engine.generate_multi directly so
    each sample becomes a standalone candidate that the selector can score.
    """
    strategies = strategies or [
        "standard_cot",
        "self_consistency",
        "backward_cloze",
        "targeted_repair",
        "random_repair",
        "full_regeneration",
    ]
    candidates: list[Candidate] = []

    prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step."

    for strat in strategies:
        if strat == "self_consistency":
            outputs = engine.generate_multi(
                prompt, n=sc_k, max_tokens=max_tokens, temperature=sc_temperature,
            )
            for i, out in enumerate(outputs):
                candidates.append(_to_candidate(
                    example_id, "sc_sample", i, out.text,
                    out.completion_tokens, out.prompt_tokens,
                    confidence=1.0 / max(1, sc_k),
                    answer_type=answer_type,
                    extra={"source_strategy": "self_consistency", "k": sc_k,
                           "temperature": sc_temperature},
                ))
            continue

        if strat not in STRATEGY_REGISTRY:
            continue

        strat_obj = build_strategy(strat)
        try:
            res: StrategyResult = strat_obj.run(
                engine, question, max_tokens=max_tokens, few_shot=few_shot,
            )
        except Exception as e:
            candidates.append(_to_candidate(
                example_id, strat, 0, "", 0, 0, confidence=0.0,
                answer_type=answer_type,
                extra={"error": repr(e)},
            ))
            continue

        candidates.append(_to_candidate(
            example_id, strat, 0, res.reasoning_trace or "",
            res.completion_tokens, res.prompt_tokens,
            confidence=float(res.confidence),
            answer_type=answer_type,
            extra={
                "strategy_prediction": res.prediction,
                "total_tokens": res.total_tokens,
                "step_metadata": res.step_metadata,
            },
        ))

    candidates = cluster_candidates(candidates)
    return candidates


def summarize_clusters(cands: list[Candidate]) -> dict[int, dict[str, Any]]:
    """For each cluster, compute size, supporting strategies, mean confidence, total tokens."""
    by_cluster: dict[int, list[Candidate]] = defaultdict(list)
    for c in cands:
        by_cluster[c.answer_cluster_id].append(c)

    summary = {}
    for cid, members in by_cluster.items():
        strategies = Counter(c.strategy for c in members)
        summary[cid] = {
            "cluster_id": cid,
            "answer": members[0].normalized_answer,
            "size": len(members),
            "strategy_support": dict(strategies),
            "mean_confidence": sum(c.confidence for c in members) / len(members),
            "total_tokens": sum(c.tokens for c in members),
            "candidate_ids": [c.candidate_id for c in members],
        }
    return summary
