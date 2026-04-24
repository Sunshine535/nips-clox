#!/usr/bin/env python3
"""CLOX-PCS portfolio experiment runner (GPT-5.5 Pro Task 9).

Two modes:
  collect  — run portfolio and save candidates per-example JSONL.
  eval     — run A/B/C control comparison on held-out split using collected
             candidates (no model inference needed if candidates are cached).

Controls:
  A (old_fragment)   — BAV / cross-vote / majority — pick per existing heuristic.
  B (portfolio_mv)   — same candidate pool as C, but majority vote only.
  C (pcs_full)       — portfolio + calibrated selector + value-of-compute gate.

Iso-budget: all three consume from same candidate pool; token cost is accounted
on-call only when C's gate opts to continue. A/B use the full collected pool,
so B is a strict upper bound on "portfolio without selection".

Usage:
    # Collect on calibration split
    python code/run_portfolio_experiment.py \
        --mode collect --split calib \
        --benchmark gsm8k --max_examples 50 --seed 11 \
        --out results/pcs/minimal/calib_candidates.jsonl

    # Evaluate A/B/C on test split
    python code/run_portfolio_experiment.py \
        --mode eval --compare A,B,C \
        --candidates results/pcs/minimal/test_candidates.jsonl \
        --selector results/pcs/selector.pkl \
        --out results/pcs/minimal/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _collect(args) -> None:
    # Lazy imports so eval mode doesn't require vLLM/torch
    from engine import VLLMEngine
    from benchmarks import load_benchmark, load_math, load_bbh
    from portfolio import run_portfolio
    from result_schema import create_run_manifest, save_per_example
    from utils import set_global_seed

    set_global_seed(args.seed)

    if args.benchmark == "math_hard":
        examples = load_math(max_examples=None, levels=[4, 5])
    elif args.benchmark == "bbh_logic":
        examples = load_bbh(subtasks=["logical_deduction_five_objects"])
    else:
        examples = load_benchmark(args.benchmark, max_examples=None)

    # Apply split if manifest provided
    if args.split_manifest and os.path.exists(args.split_manifest):
        manifest = json.load(open(args.split_manifest))
        id_set = set(manifest[f"{args.split}_ids"])
        examples = [e for e in examples if e.example_id in id_set]
    if args.max_examples:
        examples = examples[: args.max_examples]

    answer_type_map = {
        "gsm8k": "numeric", "math_hard": "numeric",
        "arc_challenge": "multiple_choice", "bbh_logic": "multiple_choice",
        "strategyqa": "boolean",
    }
    answer_type = answer_type_map.get(args.benchmark, "text")

    engine = VLLMEngine(model_path=args.model, tensor_parallel_size=args.tp, seed=args.seed)

    out_rows = []
    for i, ex in enumerate(examples):
        cands = run_portfolio(
            engine, ex.example_id, ex.question,
            answer_type=answer_type,
            strategies=args.strategies.split(","),
            sc_k=args.sc_k, max_tokens=args.max_tokens,
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
                "sc_k": args.sc_k, "max_tokens": args.max_tokens},
        command=" ".join(sys.argv),
    )
    print(f"[collect] saved {len(out_rows)} rows → {args.out}")


def _majority_vote(candidates: list[dict]) -> str:
    ans = [c["normalized_answer"] for c in candidates if c.get("normalized_answer")]
    if not ans:
        return ""
    return Counter(ans).most_common(1)[0][0]


def _old_fragment_pick(candidates: list[dict]) -> str:
    """A = old best fragment: use SC-style majority vote among sc_sample only."""
    sc = [c for c in candidates if c["strategy"] == "sc_sample"]
    if sc:
        return _majority_vote(sc)
    return _majority_vote(candidates)


def _pcs_pick(candidates: list[dict], selector_path: str,
              tau_stop: float = 0.75) -> tuple[str, dict]:
    from calibrated_selector import load_artifact, score_pool
    from compute_gate import aggregate_cluster_scores, pick_best_answer, decide

    art = load_artifact(selector_path)
    scores = score_pool(art, candidates)
    for c, s in zip(candidates, scores):
        c["calibrated_score"] = float(s)
    # single-shot pick — gate informs logging only (no further generation here)
    decision = decide(
        candidates, remaining_budget=0, next_strategy_cost=10**9,
        tau_stop=tau_stop,
    )
    best = pick_best_answer(candidates)
    return best.get("normalized_answer", ""), {
        "decision": asdict(decision),
        "calibrated_scores": scores,
    }


def _eval(args) -> None:
    from answer_extraction import check_answer_strict

    rows = [json.loads(l) for l in open(args.candidates) if l.strip()]
    arms = [a.strip() for a in args.compare.split(",")]
    per_example = []
    totals = {a: {"correct": 0, "tokens": 0} for a in arms}

    for row in rows:
        gt = row["ground_truth"]
        atype = row.get("answer_type", "text")
        total_tokens = sum(c.get("tokens", 0) for c in row["candidates"])
        ex_record = {
            "example_id": row["example_id"],
            "ground_truth": gt,
            "answer_type": atype,
            "total_tokens": total_tokens,
            "arms": {},
        }

        for arm in arms:
            if arm == "A":
                pred = _old_fragment_pick(row["candidates"])
                extra = {}
            elif arm == "B":
                pred = _majority_vote(row["candidates"])
                extra = {}
            elif arm == "C":
                pred, extra = _pcs_pick(
                    row["candidates"], args.selector, tau_stop=args.tau_stop,
                )
            else:
                raise ValueError(f"unknown arm {arm}")

            correct = check_answer_strict(pred, gt, atype)
            ex_record["arms"][arm] = {
                "prediction": pred, "correct": bool(correct), "extra": extra,
            }
            totals[arm]["correct"] += int(correct)
            totals[arm]["tokens"] += total_tokens

        per_example.append(ex_record)

    n = len(rows)
    summary = {
        "n_examples": n,
        "arms": {
            a: {
                "accuracy": totals[a]["correct"] / n if n else 0.0,
                "total_tokens": totals[a]["tokens"],
                "mean_tokens_per_example": totals[a]["tokens"] / n if n else 0.0,
            } for a in arms
        },
    }
    # McNemar-style pair differences
    pairs = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            a_correct = [ex["arms"][a]["correct"] for ex in per_example]
            b_correct = [ex["arms"][b]["correct"] for ex in per_example]
            b_only = sum(1 for x, y in zip(a_correct, b_correct) if (not x) and y)
            a_only = sum(1 for x, y in zip(a_correct, b_correct) if x and (not y))
            pairs[f"{a}_vs_{b}"] = {
                f"{a}_only_correct": a_only,
                f"{b}_only_correct": b_only,
                "both_correct": sum(1 for x, y in zip(a_correct, b_correct) if x and y),
                "both_wrong": sum(1 for x, y in zip(a_correct, b_correct) if not x and not y),
            }
    summary["pair_diff"] = pairs

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_example": per_example}, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))
    print(f"[eval] saved → {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "eval"], required=True)
    parser.add_argument("--out", required=True)

    # Collect args
    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split_manifest", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--strategies", type=str,
                        default="standard_cot,self_consistency,backward_cloze,targeted_repair,random_repair,full_regeneration")
    parser.add_argument("--sc_k", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)

    # Eval args
    parser.add_argument("--compare", type=str, default="A,B,C")
    parser.add_argument("--candidates", type=str, default="")
    parser.add_argument("--selector", type=str, default="")
    parser.add_argument("--tau_stop", type=float, default=0.75)

    args = parser.parse_args()
    if args.mode == "collect":
        _collect(args)
    else:
        _eval(args)


if __name__ == "__main__":
    main()
