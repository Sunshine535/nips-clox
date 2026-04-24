#!/usr/bin/env python3
"""Agreement-Gated Diversity (AGD): Adaptive prompt diversity for SC.

Only injects diverse prompts when base SC samples disagree.
When they agree, commits immediately (saving compute).

Per problem:
  1. Generate k_base samples from standard prompt (temp=0.7)
  2. If agreement >= threshold → commit (FAST PATH)
  3. Else → generate k_div samples from diverse prompts → joint vote

Compares: SC(k=8), AGD, Oracle-select, always-diverse.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 code/agd.py \
        --model /path/to/model \
        --benchmarks math_hard,bbh_logic,arc_challenge,strategyqa \
        --n_problems 30 --tp 4 --output results/agd/ModelName
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

if os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "True"):
    import hf_offline_patch  # noqa: F401

from benchmarks import FEW_SHOT_PROMPTS, load_benchmark
from engine import VLLMEngine, auto_tp, extract_answer
from evaluation import check_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agd")

STD_PROMPT = "Let's think step by step."
DIV_PROMPTS = [
    "First, let's understand what the problem is asking and make a plan. Then solve step by step.",
    "Let's think about what we need to find, then work backwards from the goal to the given information.",
    "Let's break this into smaller sub-problems and solve each one.",
    "Let's solve this step by step, then verify our answer by checking it against the original problem.",
]

K_BASE = 4
K_DIV = 4
THRESHOLDS = [0.5, 0.75, 1.0]  # test multiple gates


def load_benchmark_subset(name, n, seed=42):
    from meta_sweep import load_benchmark_subset as _lb
    return _lb(name, n, seed=seed)


def majority_vote(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def agreement_ratio(answers):
    if not answers:
        return 0.0
    counts = Counter(answers)
    return counts.most_common(1)[0][1] / len(answers)


def run_cell(engine, benchmark, examples, max_tokens=2048):
    fs_key = benchmark.split("_")[0] if benchmark != "math_hard" else "math"
    few_shot = FEW_SHOT_PROMPTS.get(fs_key, "")

    log.info("[%s] AGD: k_base=%d, k_div=%d, thresholds=%s",
             benchmark, K_BASE, K_DIV, THRESHOLDS)

    rows = []
    for i, ex in enumerate(examples):
        t0 = time.perf_counter()

        # Step 1: Generate k_base from standard prompt
        std_prompt = f"{few_shot}\nQuestion: {ex.question}\n{STD_PROMPT}"
        base_outputs = engine.generate_multi(
            std_prompt, n=K_BASE, max_tokens=max_tokens, temperature=0.7,
        )
        base_answers = [extract_answer(o.text) for o in base_outputs]
        base_tokens = sum(o.total_tokens for o in base_outputs)
        base_agreement = agreement_ratio(base_answers)
        base_majority = majority_vote(base_answers)

        # Step 2: Generate k_div from diverse prompts (always, for comparison)
        div_prompts_full = [
            f"{few_shot}\nQuestion: {ex.question}\n{dp}" for dp in DIV_PROMPTS
        ]
        div_outputs = engine.generate_batch(
            div_prompts_full, max_tokens=max_tokens, temperature=0.7,
        )
        div_answers = [extract_answer(o.text) for o in div_outputs]
        div_tokens = sum(o.total_tokens for o in div_outputs)

        # Also generate SC(k=8) for fair comparison (8 from same prompt)
        sc8_outputs = engine.generate_multi(
            std_prompt, n=8, max_tokens=max_tokens, temperature=0.7,
        )
        sc8_answers = [extract_answer(o.text) for o in sc8_outputs]
        sc8_tokens = sum(o.total_tokens for o in sc8_outputs)
        sc8_pred = majority_vote(sc8_answers)
        sc8_correct = check_answer(sc8_pred, ex.answer, ex.answer_type)

        # Joint vote (base + div)
        joint_answers = base_answers + div_answers
        joint_pred = majority_vote(joint_answers)
        joint_correct = check_answer(joint_pred, ex.answer, ex.answer_type)

        # AGD at each threshold
        agd_results = {}
        for thr in THRESHOLDS:
            if base_agreement >= thr:
                agd_pred = base_majority
                agd_tokens = base_tokens
                agd_path = "fast"
            else:
                agd_pred = majority_vote(joint_answers)
                agd_tokens = base_tokens + div_tokens
                agd_path = "diverse"
            agd_correct = check_answer(agd_pred, ex.answer, ex.answer_type)
            agd_results[str(thr)] = {
                "pred": agd_pred, "correct": agd_correct,
                "tokens": agd_tokens, "path": agd_path,
            }

        elapsed = time.perf_counter() - t0
        rows.append({
            "example_id": ex.example_id,
            "ground_truth": ex.answer,
            "base_agreement": base_agreement,
            "base_majority": base_majority,
            "base_answers": base_answers,
            "div_answers": div_answers,
            "sc8_correct": sc8_correct, "sc8_tokens": sc8_tokens,
            "joint_correct": joint_correct,
            "agd": agd_results,
            "elapsed": round(elapsed, 1),
        })

        if (i + 1) % 5 == 0:
            n_done = len(rows)
            sc = sum(r["sc8_correct"] for r in rows) / n_done
            parts = [f"SC8={sc:.0%}"]
            for thr in THRESHOLDS:
                a = sum(r["agd"][str(thr)]["correct"] for r in rows) / n_done
                fp = sum(1 for r in rows if r["agd"][str(thr)]["path"] == "fast") / n_done
                parts.append(f"AGD{thr}={a:.0%}(f{fp:.0%})")
            log.info("  [%s] %d/%d  %s", benchmark, n_done, len(examples), "  ".join(parts))

    n = len(rows)
    summary = {"benchmark": benchmark, "n_examples": n}

    sc_acc = sum(r["sc8_correct"] for r in rows) / n
    sc_tok = sum(r["sc8_tokens"] for r in rows) / n
    joint_acc = sum(r["joint_correct"] for r in rows) / n
    summary["sc8"] = {"accuracy": sc_acc, "mean_tokens": sc_tok}
    summary["always_diverse"] = {"accuracy": joint_acc}

    for thr in THRESHOLDS:
        acc = sum(r["agd"][str(thr)]["correct"] for r in rows) / n
        tok = sum(r["agd"][str(thr)]["tokens"] for r in rows) / n
        fast_pct = sum(1 for r in rows if r["agd"][str(thr)]["path"] == "fast") / n
        summary[f"agd_{thr}"] = {
            "accuracy": acc, "mean_tokens": tok,
            "fast_pct": fast_pct,
            "delta_vs_sc": acc - sc_acc,
            "token_ratio": tok / max(sc_tok, 1),
        }

    return {"summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmarks", type=str,
                        default="math_hard,bbh_logic,arc_challenge,strategyqa")
    parser.add_argument("--n_problems", type=int, default=30)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--quant", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.output, "agd.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    log.info("=== AGD Experiment ===")
    log.info("Model: %s, Benchmarks: %s, N=%d", args.model, benchmarks, args.n_problems)

    tp = args.tp if args.tp > 0 else auto_tp(args.model)
    engine = VLLMEngine(
        model_name=args.model, tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
        quantization=args.quant, seed=args.seed,
    )
    log.info("Model loaded (tp=%d)", tp)

    all_results = {}
    for bname in benchmarks:
        log.info("=== Benchmark: %s ===", bname)
        try:
            examples = load_benchmark_subset(bname, args.n_problems, seed=args.seed)
        except Exception as e:
            log.error("Failed to load %s: %s", bname, e)
            continue

        cell = run_cell(engine, bname, examples, max_tokens=args.max_tokens)
        all_results[bname] = cell

        with open(os.path.join(args.output, "agd_results.json"), "w") as f:
            json.dump({"model": args.model, "cells": all_results},
                      f, indent=2, default=str)

        s = cell["summary"]
        log.info("[%s] RESULTS:", bname)
        log.info("  SC(8):          %5.1f%%", s["sc8"]["accuracy"] * 100)
        log.info("  Always-diverse: %5.1f%%", s["always_diverse"]["accuracy"] * 100)
        for thr in THRESHOLDS:
            a = s[f"agd_{thr}"]
            log.info("  AGD(thr=%.2f):  %5.1f%%  Δ=%+.1f%%  fast=%.0f%%  tok=%.2fx",
                     thr, a["accuracy"]*100, a["delta_vs_sc"]*100,
                     a["fast_pct"]*100, a["token_ratio"])

    log.info("\n=== FINAL CROSS-BENCHMARK ===")
    if all_results:
        for thr in THRESHOLDS:
            deltas = [all_results[b]["summary"][f"agd_{thr}"]["delta_vs_sc"]
                      for b in all_results]
            avg = np.mean(deltas)
            min_d = min(deltas)
            log.info("AGD(%.2f): avg Δ=%+.1f%%, min Δ=%+.1f%% (%s)",
                     thr, avg*100, min_d*100,
                     "UNIVERSAL" if min_d >= 0 else "NOT universal")


if __name__ == "__main__":
    main()
