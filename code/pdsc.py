#!/usr/bin/env python3
"""Prompt-Diverse Self-Consistency (PDSC) vs Standard SC.

Core idea: SC uses k samples from ONE prompt (temperature diversity only).
PDSC distributes k samples across K different reasoning prompts,
adding strategy diversity at identical compute budget.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 code/pdsc.py \
        --model /path/to/model \
        --benchmarks strategyqa,bbh_logic,arc_challenge \
        --n_problems 30 --k 8 --tp 2 \
        --output results/pdsc/ModelName
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

if os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "True"):
    import hf_offline_patch  # noqa: F401

from benchmarks import FEW_SHOT_PROMPTS, load_benchmark
from engine import VLLMEngine, auto_tp, extract_answer
from evaluation import check_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pdsc")

REASONING_PROMPTS = [
    "Let's think step by step.",
    "First, let's understand what the problem is asking and make a plan. Then solve step by step.",
    "Let's think about what we need to find, then work backwards from the goal to the given information.",
    "Let's break this into smaller sub-problems and solve each one.",
    "Let's solve this step by step, then verify our answer by checking it against the original problem.",
    "Let's carefully list all the given information, identify what we need to find, then reason through.",
    "Let's consider this problem from multiple angles before deciding on an approach.",
    "Let's solve this by first identifying the key relationships, then computing step by step.",
]


def load_benchmark_subset(name: str, n: int, seed: int = 42) -> list:
    from meta_sweep import load_benchmark_subset as _lb
    return _lb(name, n, seed=seed)


def majority_vote(answers: list[str]) -> str:
    if not answers:
        return ""
    counts = Counter(answers)
    return counts.most_common(1)[0][0]


def run_pdsc_cell(engine, benchmark, examples, k=8, max_tokens=2048):
    fs_key = benchmark.split("_")[0] if benchmark != "math_hard" else "math"
    few_shot = FEW_SHOT_PROMPTS.get(fs_key, "")

    n_prompts = len(REASONING_PROMPTS)
    samples_per_prompt = max(1, k // n_prompts)
    prompts_to_use = REASONING_PROMPTS[:k] if k <= n_prompts else REASONING_PROMPTS
    k_pdsc = len(prompts_to_use) * samples_per_prompt

    log.info("[%s] SC(k=%d) vs PDSC(k=%d, %d prompts × %d samples)",
             benchmark, k, k_pdsc, len(prompts_to_use), samples_per_prompt)

    results = []
    for i, ex in enumerate(examples):
        t0 = time.perf_counter()

        # --- Standard SC: k samples from prompt_0 ---
        sc_prompt = f"{few_shot}\nQuestion: {ex.question}\n{REASONING_PROMPTS[0]}"
        sc_outputs = engine.generate_multi(
            sc_prompt, n=k, max_tokens=max_tokens, temperature=0.7,
        )
        sc_answers = [extract_answer(o.text) for o in sc_outputs]
        sc_pred = majority_vote(sc_answers)
        sc_tokens = sum(o.total_tokens for o in sc_outputs)
        sc_correct = check_answer(sc_pred, ex.answer, ex.answer_type)

        # --- PDSC: samples distributed across diverse prompts ---
        pdsc_all_answers = []
        pdsc_tokens = 0

        if samples_per_prompt == 1:
            batch_prompts = [
                f"{few_shot}\nQuestion: {ex.question}\n{rp}"
                for rp in prompts_to_use
            ]
            batch_outputs = engine.generate_batch(
                batch_prompts, max_tokens=max_tokens, temperature=0.7,
            )
            for o in batch_outputs:
                pdsc_all_answers.append(extract_answer(o.text))
                pdsc_tokens += o.total_tokens
        else:
            for rp in prompts_to_use:
                p = f"{few_shot}\nQuestion: {ex.question}\n{rp}"
                outs = engine.generate_multi(
                    p, n=samples_per_prompt, max_tokens=max_tokens, temperature=0.7,
                )
                for o in outs:
                    pdsc_all_answers.append(extract_answer(o.text))
                    pdsc_tokens += o.total_tokens

        pdsc_pred = majority_vote(pdsc_all_answers)
        pdsc_correct = check_answer(pdsc_pred, ex.answer, ex.answer_type)

        elapsed = time.perf_counter() - t0
        results.append({
            "example_id": ex.example_id,
            "ground_truth": ex.answer,
            "sc_pred": sc_pred, "sc_correct": sc_correct,
            "sc_answers": sc_answers, "sc_tokens": sc_tokens,
            "pdsc_pred": pdsc_pred, "pdsc_correct": pdsc_correct,
            "pdsc_answers": pdsc_all_answers, "pdsc_tokens": pdsc_tokens,
            "elapsed": round(elapsed, 1),
        })

        if (i + 1) % 5 == 0:
            sc_acc = sum(r["sc_correct"] for r in results) / len(results)
            pdsc_acc = sum(r["pdsc_correct"] for r in results) / len(results)
            log.info("  [%s] %d/%d  SC=%.3f  PDSC=%.3f  Δ=%+.3f",
                     benchmark, i + 1, len(examples), sc_acc, pdsc_acc, pdsc_acc - sc_acc)

    sc_acc = sum(r["sc_correct"] for r in results) / max(len(results), 1)
    pdsc_acc = sum(r["pdsc_correct"] for r in results) / max(len(results), 1)
    sc_mean_tok = sum(r["sc_tokens"] for r in results) / max(len(results), 1)
    pdsc_mean_tok = sum(r["pdsc_tokens"] for r in results) / max(len(results), 1)

    return {
        "benchmark": benchmark,
        "n_examples": len(examples),
        "k": k,
        "n_prompts": len(prompts_to_use),
        "samples_per_prompt": samples_per_prompt,
        "sc_accuracy": sc_acc,
        "pdsc_accuracy": pdsc_acc,
        "delta": pdsc_acc - sc_acc,
        "sc_mean_tokens": sc_mean_tok,
        "pdsc_mean_tokens": pdsc_mean_tok,
        "token_ratio": pdsc_mean_tok / max(sc_mean_tok, 1),
        "rows": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmarks", type=str,
                        default="strategyqa,bbh_logic,arc_challenge")
    parser.add_argument("--n_problems", type=int, default=30)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--quant", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.output, "pdsc.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    log.info("=== PDSC Experiment ===")
    log.info("Model: %s", args.model)
    log.info("Benchmarks: %s, K=%d, N=%d", benchmarks, args.k, args.n_problems)
    log.info("Diverse prompts: %d", len(REASONING_PROMPTS))

    tp = args.tp if args.tp > 0 else auto_tp(args.model)
    log.info("Loading model (tp=%d)...", tp)
    engine = VLLMEngine(
        model_name=args.model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        quantization=args.quant,
        seed=args.seed,
    )
    log.info("Model loaded")

    all_results = {}
    for bname in benchmarks:
        log.info("=== Benchmark: %s ===", bname)
        try:
            examples = load_benchmark_subset(bname, args.n_problems, seed=args.seed)
        except Exception as e:
            log.error("Failed to load %s: %s", bname, e)
            continue
        if not examples:
            log.warning("No examples for %s", bname)
            continue

        cell = run_pdsc_cell(engine, bname, examples, k=args.k,
                             max_tokens=args.max_tokens)
        all_results[bname] = cell

        out_path = os.path.join(args.output, "pdsc_results.json")
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "cells": all_results},
                      f, indent=2, default=str)

        log.info("[%s] RESULT: SC=%.3f  PDSC=%.3f  Δ=%+.3f  tok_ratio=%.2f",
                 bname, cell["sc_accuracy"], cell["pdsc_accuracy"],
                 cell["delta"], cell["token_ratio"])

    log.info("\n=== FINAL SUMMARY ===")
    log.info("%-15s %8s %8s %8s %8s", "Benchmark", "SC", "PDSC", "Delta", "TokRatio")
    for bname, cell in all_results.items():
        log.info("%-15s %8.3f %8.3f %+8.3f %8.2f",
                 bname, cell["sc_accuracy"], cell["pdsc_accuracy"],
                 cell["delta"], cell["token_ratio"])

    total_sc = np.mean([c["sc_accuracy"] for c in all_results.values()])
    total_pdsc = np.mean([c["pdsc_accuracy"] for c in all_results.values()])
    log.info("%-15s %8.3f %8.3f %+8.3f", "AVERAGE", total_sc, total_pdsc,
             total_pdsc - total_sc)


if __name__ == "__main__":
    main()
