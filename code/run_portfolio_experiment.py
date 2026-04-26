#!/usr/bin/env python3
"""CLOX-PCS portfolio experiment runner (GPT-5.5 Pro Tasks 1, 4, 5, 7).

Two modes:
  collect  — run portfolio and save candidates per-example JSONL.
  eval     — simulate staged collection on cached candidates: scout stage →
             selector + gate → optional expansion. C arm token cost reflects
             only the stages actually used.

Eval arms:
  A_SC        — majority vote over sc_sample candidates only (legacy SC).
  A_BAV       — BAV candidate prediction (skipped if not collected).
  A_TARGETED  — targeted_repair candidate prediction.
  B           — majority vote over the FULL candidate pool (no selector).
  C           — staged PCS: selector + active value-of-compute gate.

Usage (mock-engine smoke):
    python code/run_portfolio_experiment.py --mode collect \
        --benchmark gsm8k --max_examples 1 --seed 11 \
        --model dummy --mock_engine \
        --out /tmp/pcs_collect_smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ────────────────────────────────────────────────────────────────────
# COLLECT
# ────────────────────────────────────────────────────────────────────


def _build_engine(args):
    """Engine factory. Honours --mock_engine for tests / smoke runs."""
    if getattr(args, "mock_engine", False):
        from mock_engine import MockEngine
        return MockEngine(seed=args.seed)
    from engine import VLLMEngine
    return VLLMEngine(
        model_name=args.model,
        tensor_parallel_size=args.tp,
        seed=args.seed,
    )


def _collect(args) -> None:
    from benchmarks import load_benchmark, load_math, load_bbh
    from portfolio import run_portfolio
    from result_schema import create_run_manifest
    from utils import set_global_seed

    set_global_seed(args.seed)

    if args.benchmark == "math_hard":
        examples = load_math(max_examples=None, levels=[4, 5])
    elif args.benchmark == "bbh_logic":
        examples = load_bbh(subtasks=["logical_deduction_five_objects"])
    else:
        examples = load_benchmark(args.benchmark, max_examples=None)

    if args.split_manifest and os.path.exists(args.split_manifest):
        manifest = json.load(open(args.split_manifest))
        id_set = set(manifest[f"{args.split}_ids"])
        examples = [e for e in examples if e.example_id in id_set]
    if args.max_examples:
        examples = examples[: args.max_examples]

    answer_type_map = {
        "gsm8k": "numeric", "math_hard": "math_expression",
        "arc_challenge": "multiple_choice", "bbh_logic": "multiple_choice",
        "strategyqa": "boolean",
    }
    answer_type = answer_type_map.get(args.benchmark, "text")

    engine = _build_engine(args)

    out_rows = []
    for i, ex in enumerate(examples):
        cands = run_portfolio(
            engine, ex.example_id, ex.question,
            answer_type=answer_type,
            strategies=args.strategies.split(","),
            sc_k=args.sc_k, max_tokens=args.max_tokens,
            seed=args.seed,
        )
        row = {
            "example_id": ex.example_id,
            "question": ex.question,
            "ground_truth": ex.answer,
            "answer_type": answer_type,
            "candidates": [c.to_dict() for c in cands],
        }
        out_rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"[collect] {i+1}/{len(examples)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, default=str) + "\n")

    manifest_dir = os.path.dirname(args.out) or "."
    create_run_manifest(
        manifest_dir, model=args.model, benchmark=args.benchmark,
        split=args.split, seed=args.seed,
        config={"mode": "collect", "strategies": args.strategies,
                "sc_k": args.sc_k, "max_tokens": args.max_tokens,
                "mock_engine": bool(getattr(args, "mock_engine", False))},
        command=" ".join(sys.argv),
    )
    print(f"[collect] saved {len(out_rows)} rows → {args.out}")


# ────────────────────────────────────────────────────────────────────
# EVAL — A controls + B + C with staged active gate
# ────────────────────────────────────────────────────────────────────


def _majority_vote(candidates: list[dict]) -> str:
    ans = [c["normalized_answer"] for c in candidates if c.get("normalized_answer")]
    if not ans:
        return ""
    return Counter(ans).most_common(1)[0][0]


def _pick_strategy(candidates: list[dict], strategy: str) -> str:
    """Pick the prediction of the (first) candidate from the named strategy."""
    for c in candidates:
        if c.get("strategy") == strategy:
            return c.get("normalized_answer", "")
    return ""


def _arm_A_SC(candidates: list[dict]) -> tuple[str, int]:
    sc = [c for c in candidates if c.get("strategy") == "sc_sample"]
    pred = _majority_vote(sc) if sc else _majority_vote(candidates)
    tokens = sum(c.get("tokens", 0) for c in (sc if sc else candidates))
    return pred, tokens


def _arm_A_BAV(candidates: list[dict]) -> tuple[str, int]:
    bav = [c for c in candidates if c.get("strategy") == "bav"]
    if not bav:
        return "", 0
    return bav[0].get("normalized_answer", ""), sum(c.get("tokens", 0) for c in bav)


def _arm_A_TARGETED(candidates: list[dict]) -> tuple[str, int]:
    t = [c for c in candidates if c.get("strategy") == "targeted_repair"]
    if not t:
        return "", 0
    return t[0].get("normalized_answer", ""), sum(c.get("tokens", 0) for c in t)


def _arm_B(candidates: list[dict]) -> tuple[str, int]:
    return _majority_vote(candidates), sum(c.get("tokens", 0) for c in candidates)


SCOUT_STRATEGIES = {"standard_cot", "sc_sample"}


def _arm_C_staged(
    candidates: list[dict], selector_path: str, tau_stop: float, tau_margin: float,
) -> tuple[str, int, dict]:
    """Active staged PCS:
        Stage 1 = SCOUT_STRATEGIES candidates (cheap baseline pool).
        Score with selector → gate → if STOP, return; else add expansion candidates.
    Token accounting reflects only stages actually used.
    """
    from calibrated_selector import load_artifact, score_pool
    from compute_gate import decide, pick_best_answer
    from features import extract_features

    art = load_artifact(selector_path)

    scout = [c for c in candidates if c.get("strategy") in SCOUT_STRATEGIES]
    expansion = [c for c in candidates if c.get("strategy") not in SCOUT_STRATEGIES]
    if not scout:
        scout, expansion = candidates, []

    # Stage-1 scoring & gating
    pool = list(scout)
    scores = score_pool(art, pool)
    for c, s in zip(pool, scores):
        c["calibrated_score"] = float(s)
    scout_tokens = sum(c.get("tokens", 0) for c in scout)
    expansion_tokens = sum(c.get("tokens", 0) for c in expansion)

    decision_stage1 = decide(
        pool, remaining_budget=expansion_tokens,
        next_strategy_cost=expansion_tokens or 1,
        tau_stop=tau_stop, tau_margin=tau_margin,
    )

    common_log = {
        "mechanism_enabled": True,
        "selector_artifact_path": selector_path,
        "feature_schema_hash": art.meta.get("feature_schema_hash", ""),
        "scout_tokens": scout_tokens,
        "expansion_tokens": expansion_tokens,
        "pool_total_tokens": scout_tokens + expansion_tokens,
    }

    if decision_stage1.action == "stop" or not expansion:
        best = pick_best_answer(pool)
        return best.get("normalized_answer", ""), scout_tokens, {
            **common_log,
            "stage": 1, "decision_stage1": asdict(decision_stage1),
            "stage1_pool_size": len(pool), "expansion_used": False,
            "tokens_used": scout_tokens,
            "best_cluster_answer": best.get("normalized_answer", ""),
            "calibrated_scores": scores,
        }

    # Stage-2: expand & re-score full pool
    pool = list(scout) + list(expansion)
    scores_full = score_pool(art, pool)
    for c, s in zip(pool, scores_full):
        c["calibrated_score"] = float(s)
    decision_stage2 = decide(
        pool, remaining_budget=0, next_strategy_cost=10**9,
        tau_stop=tau_stop, tau_margin=tau_margin,
    )
    best = pick_best_answer(pool)
    return best.get("normalized_answer", ""), scout_tokens + expansion_tokens, {
        **common_log,
        "stage": 2, "decision_stage1": asdict(decision_stage1),
        "decision_stage2": asdict(decision_stage2),
        "stage1_pool_size": len(scout),
        "stage2_pool_size": len(pool),
        "expansion_used": True,
        "tokens_used": scout_tokens + expansion_tokens,
        "best_cluster_answer": best.get("normalized_answer", ""),
        "calibrated_scores": scores_full,
    }


ARM_FNS = {
    "A_SC": _arm_A_SC,
    "A_BAV": _arm_A_BAV,
    "A_TARGETED": _arm_A_TARGETED,
    "B": _arm_B,
    # Legacy aliases (backwards compat)
    "A": _arm_A_SC,
}


def _eval(args) -> None:
    from answer_extraction import check_answer_strict

    rows = [json.loads(l) for l in open(args.candidates) if l.strip()]
    arms = [a.strip() for a in args.compare.split(",")]
    per_example = []
    totals = {a: {"correct": 0, "tokens": 0, "n_voted": 0} for a in arms}

    for row in rows:
        gt = row["ground_truth"]
        atype = row.get("answer_type", "text")
        ex_record = {
            "example_id": row["example_id"],
            "ground_truth": gt,
            "answer_type": atype,
            "pool_total_tokens": sum(c.get("tokens", 0) for c in row["candidates"]),
            "arms": {},
        }

        for arm in arms:
            if arm == "C":
                pred, tokens, extra = _arm_C_staged(
                    row["candidates"], args.selector,
                    tau_stop=args.tau_stop, tau_margin=args.tau_margin,
                )
            elif arm in ARM_FNS:
                pred, tokens = ARM_FNS[arm](row["candidates"])
                extra = {}
            else:
                raise ValueError(f"unknown arm: {arm}. Available: {sorted(ARM_FNS)} + C")

            voted = bool(pred)  # arm produced a usable answer
            correct = bool(check_answer_strict(pred, gt, atype)) if voted else False
            ex_record["arms"][arm] = {
                "prediction": pred, "correct": correct,
                "tokens_used": int(tokens), "voted": voted, "extra": extra,
            }
            totals[arm]["correct"] += int(correct)
            totals[arm]["tokens"] += int(tokens)
            totals[arm]["n_voted"] += int(voted)

        per_example.append(ex_record)

    n = len(rows)
    summary = {
        "n_examples": n,
        "arms": {
            a: {
                "accuracy": totals[a]["correct"] / n if n else 0.0,
                "n_voted": totals[a]["n_voted"],
                "total_tokens": totals[a]["tokens"],
                "mean_tokens_per_example": totals[a]["tokens"] / n if n else 0.0,
                "tokens_per_correct": (totals[a]["tokens"] / totals[a]["correct"])
                                      if totals[a]["correct"] else float("inf"),
            } for a in arms
        },
    }
    pairs = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            ac = [ex["arms"][a]["correct"] for ex in per_example]
            bc = [ex["arms"][b]["correct"] for ex in per_example]
            pairs[f"{a}_vs_{b}"] = {
                f"{a}_only_correct": sum(1 for x, y in zip(ac, bc) if x and not y),
                f"{b}_only_correct": sum(1 for x, y in zip(ac, bc) if y and not x),
                "both_correct": sum(1 for x, y in zip(ac, bc) if x and y),
                "both_wrong": sum(1 for x, y in zip(ac, bc) if not x and not y),
            }
    summary["pair_diff"] = pairs

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_example": per_example},
                  f, indent=2, default=str)
    print(json.dumps(summary, indent=2))
    print(f"[eval] saved → {args.out}")


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "eval"], required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split_manifest", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--mock_engine", action="store_true",
                        help="Skip vLLM; use code/mock_engine.py for tests/smoke.")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--strategies", type=str,
                        default="standard_cot,self_consistency,backward_cloze,targeted_repair,random_repair,full_regeneration")
    parser.add_argument("--sc_k", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--allow_train_eval", action="store_true",
                        help="Permit StrategyQA train-split fallback (default forbidden).")

    parser.add_argument("--compare", type=str, default="A_SC,A_TARGETED,A_BAV,B,C")
    parser.add_argument("--candidates", type=str, default="")
    parser.add_argument("--selector", type=str, default="")
    parser.add_argument("--tau_stop", type=float, default=0.75)
    parser.add_argument("--tau_margin", type=float, default=0.10)

    args = parser.parse_args()
    if args.mode == "collect":
        _collect(args)
    else:
        _eval(args)


if __name__ == "__main__":
    main()
