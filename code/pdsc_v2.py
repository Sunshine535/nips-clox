#!/usr/bin/env python3
"""PDSC v2: Depth-Breadth Ablation + Confidence-Weighted Voting.

Generates ALL samples in one pass per problem, then evaluates every
(depth, breadth, voting) configuration offline. Much more efficient
than running each config separately.

Per problem generates:
  - 8 samples from standard prompt (temperature=0.7)
  - 1 sample from each of 7 diverse prompts (temperature=0.7)
  Total: 15 samples → construct any hybrid from these

Configurations evaluated:
  SC:    8 std + 0 div (baseline)
  H62:   6 std + 2 div
  H53:   5 std + 3 div
  H44:   4 std + 4 div
  H35:   3 std + 5 div
  PDSC:  1 std + 7 div (= 8 total, pure diverse)

Each config tested with both equal-weight and confidence-weighted voting.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 code/pdsc_v2.py \
        --model /path/to/model \
        --benchmarks math_hard,bbh_logic,arc_challenge,strategyqa \
        --n_problems 30 --tp 4 --output results/pdsc_v2/ModelName
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
log = logging.getLogger("pdsc2")

PROMPTS = [
    "Let's think step by step.",
    "First, let's understand what the problem is asking and make a plan. Then solve step by step.",
    "Let's think about what we need to find, then work backwards from the goal to the given information.",
    "Let's break this into smaller sub-problems and solve each one.",
    "Let's solve this step by step, then verify our answer by checking it against the original problem.",
    "Let's carefully list all the given information, identify what we need to find, then reason through.",
    "Let's consider this problem from multiple angles before deciding on an approach.",
    "Let's solve this by first identifying the key relationships, then computing step by step.",
]

CONFIGS = [
    ("SC",   8, 0),
    ("H62",  6, 2),
    ("H53",  5, 3),
    ("H44",  4, 4),
    ("H35",  3, 5),
    ("H26",  2, 6),
    ("PDSC", 1, 7),
]


def load_benchmark_subset(name, n, seed=42):
    from meta_sweep import load_benchmark_subset as _lb
    return _lb(name, n, seed=seed)


def confidence_of(output):
    if output.logprobs and len(output.logprobs) > 0:
        return math.exp(sum(output.logprobs) / len(output.logprobs))
    return 0.5


def majority_vote(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def weighted_vote(answer_conf_pairs):
    if not answer_conf_pairs:
        return ""
    scores = defaultdict(float)
    for ans, conf in answer_conf_pairs:
        scores[ans] += conf
    return max(scores, key=scores.get)


def eval_config(std_outputs, div_outputs, n_std, n_div, gt, ans_type):
    samples = std_outputs[:n_std] + div_outputs[:n_div]
    answers = [s["answer"] for s in samples]
    confs = [(s["answer"], s["confidence"]) for s in samples]
    tokens = sum(s["tokens"] for s in samples)

    eq_pred = majority_vote(answers)
    cw_pred = weighted_vote(confs)
    eq_correct = check_answer(eq_pred, gt, ans_type)
    cw_correct = check_answer(cw_pred, gt, ans_type)

    return {
        "eq_pred": eq_pred, "eq_correct": eq_correct,
        "cw_pred": cw_pred, "cw_correct": cw_correct,
        "tokens": tokens,
    }


def run_cell(engine, benchmark, examples, max_tokens=2048):
    fs_key = benchmark.split("_")[0] if benchmark != "math_hard" else "math"
    few_shot = FEW_SHOT_PROMPTS.get(fs_key, "")

    log.info("[%s] Generating 8 std + 7 div = 15 samples per problem", benchmark)

    rows = []
    config_acc = {name: {"eq": 0, "cw": 0, "tok": 0} for name, _, _ in CONFIGS}

    for i, ex in enumerate(examples):
        t0 = time.perf_counter()

        std_prompt = f"{few_shot}\nQuestion: {ex.question}\n{PROMPTS[0]}"
        std_outs = engine.generate_multi(
            std_prompt, n=8, max_tokens=max_tokens, temperature=0.7,
        )
        std_samples = [{
            "answer": extract_answer(o.text),
            "confidence": confidence_of(o),
            "tokens": o.total_tokens,
            "prompt_idx": 0,
        } for o in std_outs]

        div_prompts = [
            f"{few_shot}\nQuestion: {ex.question}\n{PROMPTS[j]}"
            for j in range(1, 8)
        ]
        div_outs = engine.generate_batch(
            div_prompts, max_tokens=max_tokens, temperature=0.7,
        )
        div_samples = [{
            "answer": extract_answer(o.text),
            "confidence": confidence_of(o),
            "tokens": o.total_tokens,
            "prompt_idx": j + 1,
        } for j, o in enumerate(div_outs)]

        row = {"example_id": ex.example_id, "ground_truth": ex.answer, "configs": {}}
        for name, n_std, n_div in CONFIGS:
            r = eval_config(std_samples, div_samples, n_std, n_div,
                            ex.answer, ex.answer_type)
            row["configs"][name] = r
            config_acc[name]["eq"] += r["eq_correct"]
            config_acc[name]["cw"] += r["cw_correct"]
            config_acc[name]["tok"] += r["tokens"]

        elapsed = time.perf_counter() - t0
        row["elapsed"] = round(elapsed, 1)
        rows.append(row)

        if (i + 1) % 5 == 0:
            n_done = len(rows)
            parts = []
            for name, _, _ in CONFIGS:
                eq = config_acc[name]["eq"] / n_done
                cw = config_acc[name]["cw"] / n_done
                parts.append(f"{name}={eq:.0%}/{cw:.0%}")
            log.info("  [%s] %d/%d  %s", benchmark, n_done, len(examples),
                     "  ".join(parts))

    n = len(rows)
    summary = {}
    for name, n_std, n_div in CONFIGS:
        eq_acc = config_acc[name]["eq"] / max(n, 1)
        cw_acc = config_acc[name]["cw"] / max(n, 1)
        mean_tok = config_acc[name]["tok"] / max(n, 1)
        sc_tok = config_acc["SC"]["tok"] / max(n, 1)
        summary[name] = {
            "n_std": n_std, "n_div": n_div,
            "eq_accuracy": eq_acc, "cw_accuracy": cw_acc,
            "mean_tokens": mean_tok,
            "tok_ratio": mean_tok / max(sc_tok, 1),
        }

    return {"benchmark": benchmark, "n_examples": n, "configs": summary, "rows": rows}


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
    fh = logging.FileHandler(os.path.join(args.output, "pdsc_v2.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    log.info("=== PDSC v2: Depth-Breadth Ablation + Confidence Weighting ===")
    log.info("Model: %s, Benchmarks: %s, N=%d", args.model, benchmarks, args.n_problems)
    log.info("Configs: %s", [(n, s, d) for n, s, d in CONFIGS])

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

        with open(os.path.join(args.output, "pdsc_v2_results.json"), "w") as f:
            json.dump({"model": args.model, "cells": all_results},
                      f, indent=2, default=str)

        log.info("[%s] RESULTS:", bname)
        log.info("  %-6s %8s %8s %8s %8s", "Config", "EqVote", "CWVote", "TokRatio", "k")
        sc_eq = cell["configs"]["SC"]["eq_accuracy"]
        for name, n_std, n_div in CONFIGS:
            c = cell["configs"][name]
            delta_eq = c["eq_accuracy"] - sc_eq
            delta_cw = c["cw_accuracy"] - sc_eq
            log.info("  %-6s %7.1f%% %7.1f%% %8.2f  %d+%d  (Δeq=%+.1f%% Δcw=%+.1f%%)",
                     name, c["eq_accuracy"]*100, c["cw_accuracy"]*100,
                     c["tok_ratio"], n_std, n_div, delta_eq*100, delta_cw*100)

    log.info("\n=== CROSS-BENCHMARK SUMMARY ===")
    if all_results:
        log.info("%-15s " + " ".join(f"{'Δeq':>7} {'Δcw':>7}" for _ in all_results), "Config",)
        for name, _, _ in CONFIGS:
            parts = [name]
            for bname, cell in all_results.items():
                sc_eq = cell["configs"]["SC"]["eq_accuracy"]
                deq = cell["configs"][name]["eq_accuracy"] - sc_eq
                dcw = cell["configs"][name]["cw_accuracy"] - sc_eq
                parts.append(f"{deq:>+7.1%} {dcw:>+7.1%}")
            log.info("  ".join(parts))

    # Find best config: highest min-delta-cw across benchmarks
    best_name, best_min = "SC", 0.0
    for name, _, _ in CONFIGS:
        deltas = []
        for cell in all_results.values():
            sc_eq = cell["configs"]["SC"]["eq_accuracy"]
            deltas.append(cell["configs"][name]["cw_accuracy"] - sc_eq)
        min_d = min(deltas) if deltas else -1
        if min_d > best_min:
            best_min = min_d
            best_name = name
    log.info("Best universally-positive config: %s (min Δcw = %+.1f%%)",
             best_name, best_min * 100)


if __name__ == "__main__":
    main()
