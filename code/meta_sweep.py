#!/usr/bin/env python3
"""Meta-sweep: measure Oracle−SC gap per (model, benchmark) cell.

Only 3 strategies per cell to minimize compute:
  - standard_cot
  - self_consistency (k=8)
  - backward_cloze

For each cell, compute:
  - SC accuracy
  - Oracle(any-of-3 correct) accuracy
  - Gap = Oracle − SC
  - Per-strategy accuracy

Goal: find at least one cell with Oracle−SC ≥ 15 percentage points.
That cell is where cross-strategy methods can plausibly beat SC.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 code/meta_sweep.py \\
        --model /openbayes/input/input0/hub/Qwen2.5-7B \\
        --benchmarks gsm8k,math,strategyqa,arc_challenge \\
        --n_problems 30 --tp 2 --output results/meta/Qwen2.5-7B
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# Offline HF patch before vllm imports
if os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "True"):
    import hf_offline_patch  # noqa: F401

from benchmarks import FEW_SHOT_PROMPTS, load_benchmark
from engine import VLLMEngine, auto_tp
from evaluation import check_answer
from strategies_v2 import build_strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("meta")

STRATEGIES = ["standard_cot", "self_consistency", "backward_cloze"]
STRATEGY_KWARGS = {"self_consistency": {"k": 8}}


def select_balanced(examples: list, n: int, seed: int = 42) -> list:
    """Deterministically pick n examples by stratified sampling on difficulty."""
    rng = np.random.default_rng(seed)
    if not examples:
        return []
    by_diff: dict[str, list] = {}
    for ex in examples:
        by_diff.setdefault(ex.difficulty, []).append(ex)
    # Target uniform-ish across difficulties
    ndiff = len(by_diff)
    per = max(1, n // ndiff)
    selected = []
    for diff in sorted(by_diff):
        pool = by_diff[diff]
        k = min(per, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        selected.extend(pool[i] for i in sorted(idx))
    # If short, fill from remaining
    if len(selected) < n:
        remaining_ids = {e.example_id for e in selected}
        extras = [e for e in examples if e.example_id not in remaining_ids]
        k = min(n - len(selected), len(extras))
        if k > 0:
            idx = rng.choice(len(extras), size=k, replace=False)
            selected.extend(extras[i] for i in sorted(idx))
    rng.shuffle(selected)
    return selected[:n]


def load_benchmark_subset(name: str, n: int, seed: int = 42) -> list:
    """Load a benchmark with MATH Level 4-5 filter and balanced sampling."""
    from benchmarks import load_math, load_benchmark as _lb
    if name == "math_hard":
        examples = load_math(max_examples=None, levels=[4, 5])
    elif name == "bbh_logic":
        from benchmarks import load_bbh
        examples = load_bbh(subtasks=["logical_deduction_five_objects"])
    else:
        examples = _lb(name, max_examples=None)
    return select_balanced(examples, n, seed=seed)


def run_cell(engine: VLLMEngine, benchmark: str, examples: list, max_tokens: int = 2048) -> dict:
    """Run 3 strategies on one benchmark. Return per-strategy + oracle results."""
    # Pick few-shot prompt
    fs_key = benchmark.split("_")[0] if benchmark != "math_hard" else "math"
    few_shot = FEW_SHOT_PROMPTS.get(fs_key, "")

    per_strategy: dict[str, list[dict]] = {}
    for sname in STRATEGIES:
        log.info("  [%s] strategy=%s: %d examples", benchmark, sname, len(examples))
        strat = build_strategy(sname, **STRATEGY_KWARGS.get(sname, {}))
        rows = []
        t0 = time.perf_counter()
        for i, ex in enumerate(examples):
            try:
                r = strat.run(engine, ex.question, max_tokens=max_tokens, few_shot=few_shot)
                correct = check_answer(r.prediction, ex.answer, ex.answer_type)
                rows.append({
                    "example_id": ex.example_id,
                    "correct": correct,
                    "prediction": r.prediction,
                    "ground_truth": ex.answer,
                    "total_tokens": r.total_tokens,
                })
            except Exception as e:
                log.warning("  [%s] error on %s: %s", benchmark, ex.example_id, str(e)[:100])
                rows.append({
                    "example_id": ex.example_id, "correct": False,
                    "error": str(e)[:200], "total_tokens": 0,
                })
            if (i + 1) % 10 == 0:
                acc = sum(r.get("correct", False) for r in rows) / len(rows)
                log.info("    [%s/%s] %d/%d acc=%.3f", benchmark, sname, i + 1, len(examples), acc)
        t = time.perf_counter() - t0
        acc = sum(r.get("correct", False) for r in rows) / max(len(rows), 1)
        log.info("  [%s] %s done: acc=%.3f in %.1f min", benchmark, sname, acc, t / 60)
        per_strategy[sname] = rows

    # Oracle: any strategy correct
    n = len(examples)
    oracle_correct = []
    for i, ex in enumerate(examples):
        any_correct = any(
            per_strategy[s][i].get("correct", False) for s in STRATEGIES if i < len(per_strategy[s])
        )
        oracle_correct.append(any_correct)
    oracle_acc = sum(oracle_correct) / max(n, 1)
    sc_acc = sum(r.get("correct", False) for r in per_strategy["self_consistency"]) / max(n, 1)

    return {
        "benchmark": benchmark,
        "n_examples": n,
        "per_strategy_accuracy": {
            s: sum(r.get("correct", False) for r in per_strategy[s]) / max(n, 1)
            for s in STRATEGIES
        },
        "per_strategy_mean_tokens": {
            s: float(np.mean([r.get("total_tokens", 0) for r in per_strategy[s] if r.get("total_tokens", 0) > 0]))
            for s in STRATEGIES
        },
        "oracle_accuracy": oracle_acc,
        "sc_accuracy": sc_acc,
        "oracle_sc_gap": oracle_acc - sc_acc,
        "per_strategy_rows": per_strategy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmarks", type=str,
                        default="gsm8k,math_hard,strategyqa,bbh_logic")
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
    fh = logging.FileHandler(os.path.join(args.output, "sweep.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    log.info("=== Meta-Sweep ===")
    log.info("Model: %s", args.model)
    log.info("Benchmarks: %s", benchmarks)
    log.info("N per cell: %d, Strategies: %s", args.n_problems, STRATEGIES)

    # Load engine
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

    # Run each benchmark (merge with existing results if any)
    out_path = os.path.join(args.output, "sweep_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        log.info("Loaded %d existing cells from %s", len(results.get("cells", {})), out_path)
    else:
        results = {"model": args.model, "cells": {}}
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
        log.info("Loaded %d examples for %s", len(examples), bname)
        cell = run_cell(engine, bname, examples, max_tokens=args.max_tokens)
        results["cells"][bname] = cell
        # Save after each benchmark (checkpoint)
        with open(os.path.join(args.output, "sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info("[%s] SUMMARY: Oracle=%.3f, SC=%.3f, Gap=%.3f",
                 bname, cell["oracle_accuracy"], cell["sc_accuracy"], cell["oracle_sc_gap"])

    log.info("\n=== FINAL SUMMARY ===")
    log.info("%-15s %-8s %-8s %-8s", "Benchmark", "Oracle", "SC", "Gap")
    for bname, cell in results["cells"].items():
        log.info("%-15s %8.3f %8.3f %+8.3f",
                 bname, cell["oracle_accuracy"], cell["sc_accuracy"], cell["oracle_sc_gap"])


if __name__ == "__main__":
    main()
