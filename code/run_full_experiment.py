#!/usr/bin/env python3
"""CLOX v2: Full NeurIPS experiment — 2×H100 TP=2.

Runs the complete experiment pipeline:
  Phase 0: Pilot (50 examples per benchmark, identify which benchmarks separate)
  Phase 1: Topology characterization (200 examples, 5 pilot traces)
  Phase 2: Strategy comparison (full benchmark, 3 seeds)
  Phase 3: CLOX-Adaptive evaluation
  Phase 4: Statistical analysis

Usage:
    python run_full_experiment.py --phase all
    python run_full_experiment.py --phase pilot
    python run_full_experiment.py --phase strategies --benchmarks gsm8k,math
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from benchmarks import BenchmarkExample, FEW_SHOT_PROMPTS, load_benchmark
from engine import VLLMEngine, extract_answer, split_into_steps
from evaluation import (
    check_answer,
    paired_bootstrap_ci,
    mcnemar_test,
    cohens_d,
    compute_token_efficiency,
)
from strategies_v2 import STRATEGY_REGISTRY, StrategyResult, build_strategy
from topology_v2 import (
    TopologyProfile,
    batch_estimate_topology,
    estimate_topology,
    analyze_trace,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("clox_full")

PYTHON = sys.executable
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "v3")

# Core strategies for main comparison
CORE_STRATEGIES = [
    "standard_cot",
    "self_consistency",      # k=8
    "compute_matched_sc",    # k=2 (budget-matched to 2-pass strategies)
    "targeted_repair",
    "random_repair",
    "backward_cloze",
    "full_regeneration",
    "hierarchical_repair",
    "clox_adaptive",
]

SEEDS = [11, 23, 37]


def init_engine(model_name: str, tp: int = 2, quant: str = None) -> VLLMEngine:
    log.info(f"Loading {model_name} with TP={tp}...")
    engine = VLLMEngine(
        model_name=model_name,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        quantization=quant,
    )
    log.info("Model loaded!")
    return engine


# ── Checkpoint helpers ─────────────────────────────────────────────

def _ckpt_path(output_dir: str, tag: str) -> str:
    return os.path.join(output_dir, f".ckpt_{tag}.json")


def load_ckpt(path: str) -> list[dict]:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_ckpt(path: str, data: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, path)


# ── Phase 0: Pilot ────────────────────────────────────────────────

def run_pilot(engine: VLLMEngine, output_dir: str, n_pilot: int = 50):
    """Quick pilot: 50 examples × core strategies to identify separation."""
    log.info("=== Phase 0: Pilot ===")
    benchmarks = ["gsm8k", "math", "strategyqa", "arc_challenge"]
    pilot_strategies = ["standard_cot", "self_consistency", "targeted_repair",
                        "backward_cloze", "full_regeneration"]

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for bname in benchmarks:
        log.info(f"\n--- Pilot: {bname} ---")
        examples = load_benchmark(bname, max_examples=n_pilot)
        few_shot = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")
        bm_results = {}

        for sname in pilot_strategies:
            kwargs = {"k": 5} if sname == "self_consistency" else {}
            strategy = build_strategy(sname, **kwargs)
            correct = 0
            total_tokens = 0

            for ex in examples:
                try:
                    result = strategy.run(engine, ex.question,
                                         max_tokens=512, few_shot=few_shot)
                    if check_answer(result.prediction, ex.answer, ex.answer_type):
                        correct += 1
                    total_tokens += result.total_tokens
                except Exception as e:
                    log.warning(f"  Error {sname}/{ex.example_id}: {e}")

            acc = correct / max(len(examples), 1)
            mean_tok = total_tokens / max(len(examples), 1)
            bm_results[sname] = {"accuracy": acc, "mean_tokens": mean_tok, "n": len(examples)}
            log.info(f"  {sname}: {acc:.3f} ({mean_tok:.0f} tokens)")

        results[bname] = bm_results

        # Check method separation
        accs = [v["accuracy"] for v in bm_results.values()]
        spread = max(accs) - min(accs)
        log.info(f"  Spread: {spread:.3f} (keep if > 0.03)")

    with open(os.path.join(output_dir, "pilot_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Recommend benchmarks
    recommended = []
    for bname, bm in results.items():
        accs = [v["accuracy"] for v in bm.values()]
        spread = max(accs) - min(accs)
        if spread > 0.03:
            recommended.append(bname)
    log.info(f"\nRecommended benchmarks (spread > 3%): {recommended}")
    return results, recommended


# ── Phase 1: Topology ─────────────────────────────────────────────

def run_topology(engine: VLLMEngine, benchmarks: list[str],
                 output_dir: str, n_examples: int = 200, n_pilot: int = 5):
    """Topology characterization for selected benchmarks."""
    log.info("=== Phase 1: Topology Characterization ===")
    os.makedirs(output_dir, exist_ok=True)
    summaries = {}

    for bname in benchmarks:
        log.info(f"\n--- Topology: {bname} ---")
        examples = load_benchmark(bname, max_examples=n_examples)
        few_shot = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")
        questions = [ex.question for ex in examples]

        # Batch topology estimation
        batch_size = 16
        all_profiles = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            log.info(f"  Batch {i // batch_size + 1}/{(len(questions) + batch_size - 1) // batch_size}")
            profiles = batch_estimate_topology(
                engine, batch, few_shot=few_shot,
                n_pilot=n_pilot, max_tokens=512,
            )
            all_profiles.extend(profiles)

        per_example = []
        for ex, prof in zip(examples, all_profiles):
            per_example.append({
                "example_id": ex.example_id,
                "difficulty": ex.difficulty,
                "r_bar": prof.r_bar,
                "epl": prof.epl,
                "n_steps": prof.n_steps,
                "strategy": prof.strategy,
            })

        r_bars = [p["r_bar"] for p in per_example]
        epls = [p["epl"] for p in per_example]
        strats = [p["strategy"] for p in per_example]
        strat_dist = {s: strats.count(s) for s in set(strats)}

        summary = {
            "benchmark": bname,
            "n_examples": len(per_example),
            "r_bar_mean": float(np.mean(r_bars)),
            "r_bar_std": float(np.std(r_bars)),
            "r_bar_median": float(np.median(r_bars)),
            "epl_mean": float(np.mean(epls)),
            "epl_std": float(np.std(epls)),
            "epl_median": float(np.median(epls)),
            "strategy_distribution": strat_dist,
        }
        summaries[bname] = summary

        log.info(f"  r̄={summary['r_bar_mean']:.3f}±{summary['r_bar_std']:.3f}")
        log.info(f"  ℓ={summary['epl_mean']:.2f}±{summary['epl_std']:.2f}")
        log.info(f"  Strategy dist: {strat_dist}")

        bm_dir = os.path.join(output_dir, bname)
        os.makedirs(bm_dir, exist_ok=True)
        with open(os.path.join(bm_dir, "topology.json"), "w") as f:
            json.dump({"summary": summary, "per_example": per_example}, f, indent=2)

    with open(os.path.join(output_dir, "topology_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    return summaries


# ── Phase 2: Strategy Comparison ──────────────────────────────────

def run_strategies(engine: VLLMEngine, benchmarks: list[str],
                   strategy_names: list[str], seeds: list[int],
                   output_dir: str, max_examples: int = None,
                   max_tokens: int = 512):
    """Run all strategies on all benchmarks with multiple seeds."""
    log.info("=== Phase 2: Strategy Comparison ===")
    os.makedirs(output_dir, exist_ok=True)

    for bname in benchmarks:
        log.info(f"\n{'='*60}")
        log.info(f"Benchmark: {bname}")

        max_ex = max_examples
        if bname == "math" and max_ex is None:
            max_ex = 500
        examples = load_benchmark(bname, max_examples=max_ex)
        few_shot = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")
        log.info(f"Loaded {len(examples)} examples")

        bm_dir = os.path.join(output_dir, bname)
        os.makedirs(bm_dir, exist_ok=True)

        for sname in strategy_names:
            kwargs = {}
            if sname == "self_consistency":
                kwargs = {"k": 5}

            for seed in seeds:
                ckpt = _ckpt_path(bm_dir, f"{sname}_s{seed}")
                cached = load_ckpt(ckpt)
                done_ids = {r["example_id"] for r in cached}
                remaining = [ex for ex in examples if ex.example_id not in done_ids]

                if not remaining and cached:
                    acc = sum(r.get("correct", False) for r in cached) / max(len(cached), 1)
                    log.info(f"  {sname}/s{seed}: already done ({acc:.4f})")
                    continue

                if cached:
                    log.info(f"  Resuming {sname}/s{seed}: {len(cached)} done, {len(remaining)} left")

                strategy = build_strategy(sname, **kwargs)
                results = list(cached)

                for i, ex in enumerate(remaining):
                    t0 = time.perf_counter()
                    try:
                        result = strategy.run(engine, ex.question,
                                              max_tokens=max_tokens, few_shot=few_shot)
                        elapsed = (time.perf_counter() - t0) * 1000
                        correct = check_answer(result.prediction, ex.answer, ex.answer_type)

                        results.append({
                            "example_id": ex.example_id,
                            "benchmark": ex.benchmark,
                            "difficulty": ex.difficulty,
                            "prediction": result.prediction,
                            "ground_truth": ex.answer,
                            "correct": correct,
                            "confidence": result.confidence,
                            "total_tokens": result.total_tokens,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "strategy": result.strategy_name,
                            "elapsed_ms": elapsed,
                            "seed": seed,
                        })
                    except Exception as e:
                        log.warning(f"  Error {sname}/{ex.example_id}: {e}")
                        results.append({
                            "example_id": ex.example_id,
                            "correct": False,
                            "total_tokens": 0,
                            "error": str(e),
                            "seed": seed,
                        })

                    if (i + 1) % 50 == 0:
                        acc = sum(r.get("correct", False) for r in results) / len(results)
                        log.info(f"    {sname}/s{seed}: {len(results)}/{len(examples)} acc={acc:.4f}")
                        save_ckpt(ckpt, results)

                save_ckpt(ckpt, results)
                acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
                tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens", 0) > 0]
                log.info(f"  {sname}/s{seed}: acc={acc:.4f} tokens={np.mean(tokens):.0f}")

        # Aggregate results for this benchmark
        _aggregate_benchmark(bm_dir, bname, strategy_names, seeds, examples)


def _aggregate_benchmark(bm_dir: str, bname: str,
                         strategy_names: list[str], seeds: list[int],
                         examples: list):
    """Compute aggregate statistics for a benchmark."""
    aggregate = {}
    all_results = {}

    for sname in strategy_names:
        all_results[sname] = {}
        seed_accs = []
        all_correct = []
        all_tokens = []

        for seed in seeds:
            ckpt = _ckpt_path(bm_dir, f"{sname}_s{seed}")
            results = load_ckpt(ckpt)
            all_results[sname][seed] = results
            correct = [r.get("correct", False) for r in results]
            acc = sum(correct) / max(len(correct), 1)
            seed_accs.append(acc)
            all_correct.extend(correct)
            all_tokens.extend(r.get("total_tokens", 0) for r in results
                             if r.get("total_tokens", 0) > 0)

        aggregate[sname] = {
            "mean_accuracy": float(np.mean(seed_accs)) if seed_accs else 0.0,
            "std_accuracy": float(np.std(seed_accs)) if len(seed_accs) > 1 else 0.0,
            "per_seed_accuracy": {s: a for s, a in zip(seeds, seed_accs)},
            "n_total": len(all_correct),
            "n_correct": sum(all_correct),
            "mean_tokens": float(np.mean(all_tokens)) if all_tokens else 0.0,
        }

    # Pairwise statistics vs baseline
    baseline = "standard_cot"
    if baseline in all_results:
        pairwise = {}
        for sname in strategy_names:
            if sname == baseline:
                continue
            s_correct = []
            b_correct = []
            for seed in seeds:
                s_res = all_results.get(sname, {}).get(seed, [])
                b_res = all_results.get(baseline, {}).get(seed, [])
                s_correct.extend(r.get("correct", False) for r in s_res)
                b_correct.extend(r.get("correct", False) for r in b_res)

            if s_correct and b_correct and len(s_correct) == len(b_correct):
                boot = paired_bootstrap_ci(s_correct, b_correct)
                mcn = mcnemar_test(s_correct, b_correct)
                cd = cohens_d(
                    [1.0 if c else 0.0 for c in s_correct],
                    [1.0 if c else 0.0 for c in b_correct],
                )
                pairwise[f"{sname}_vs_{baseline}"] = {
                    "bootstrap": boot,
                    "mcnemar": mcn,
                    "cohens_d": cd,
                }
        aggregate["pairwise_vs_baseline"] = pairwise

    # Save
    out = {
        "benchmark": bname,
        "strategies": strategy_names,
        "seeds": seeds,
        "n_examples_per_seed": len(examples),
        "aggregate": aggregate,
    }
    with open(os.path.join(bm_dir, "strategies.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Print summary
    log.info(f"\n--- {bname} Summary ---")
    log.info(f"{'Strategy':<25s} {'Acc':>8s} {'±Std':>8s} {'Tokens':>8s}")
    log.info("-" * 52)
    for sname in strategy_names:
        a = aggregate.get(sname, {})
        log.info(f"{sname:<25s} {a.get('mean_accuracy',0):>8.4f} "
                 f"{a.get('std_accuracy',0):>8.4f} {a.get('mean_tokens',0):>8.0f}")

    # Clean checkpoints
    for p in Path(bm_dir).glob(".ckpt_*.json"):
        p.unlink(missing_ok=True)


# ── Phase 3: Topology Proxy Validation ────────────────────────────

def run_proxy_validation(engine: VLLMEngine, benchmarks: list[str],
                         output_dir: str, n_gt: int = 50, n_pilot_list=None):
    """Validate topology estimates against ground-truth (high-sample) estimates."""
    if n_pilot_list is None:
        n_pilot_list = [3, 5, 8]

    log.info("=== Phase 3: Topology Proxy Validation ===")
    os.makedirs(output_dir, exist_ok=True)
    validation = {}

    for bname in benchmarks:
        log.info(f"\n--- Proxy validation: {bname} ---")
        examples = load_benchmark(bname, max_examples=n_gt)
        few_shot = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")

        # Ground truth: 30 pilot traces
        log.info(f"  Computing ground-truth topology (n_pilot=30)...")
        gt_profiles = batch_estimate_topology(
            engine, [ex.question for ex in examples],
            few_shot=few_shot, n_pilot=30, max_tokens=512,
        )
        gt_r_bars = [p.r_bar for p in gt_profiles]
        gt_epls = [p.epl for p in gt_profiles]

        pilot_results = {}
        for m in n_pilot_list:
            log.info(f"  Computing pilot estimates (M={m})...")
            pilot_profiles = batch_estimate_topology(
                engine, [ex.question for ex in examples],
                few_shot=few_shot, n_pilot=m, max_tokens=512,
            )
            pilot_r_bars = [p.r_bar for p in pilot_profiles]
            pilot_epls = [p.epl for p in pilot_profiles]

            # Rank-order correlation
            from scipy.stats import spearmanr
            rho_r, p_r = spearmanr(gt_r_bars, pilot_r_bars)
            rho_l, p_l = spearmanr(gt_epls, pilot_epls)

            pilot_results[m] = {
                "r_bar_mean": float(np.mean(pilot_r_bars)),
                "r_bar_std": float(np.std(pilot_r_bars)),
                "epl_mean": float(np.mean(pilot_epls)),
                "epl_std": float(np.std(pilot_epls)),
                "rho_r_bar": float(rho_r),
                "rho_r_p": float(p_r),
                "rho_epl": float(rho_l),
                "rho_epl_p": float(p_l),
                "r_bar_mae": float(np.mean(np.abs(np.array(gt_r_bars) - np.array(pilot_r_bars)))),
                "epl_mae": float(np.mean(np.abs(np.array(gt_epls) - np.array(pilot_epls)))),
            }
            log.info(f"    M={m}: ρ(r̄)={rho_r:.3f} ρ(ℓ)={rho_l:.3f}")

        validation[bname] = {
            "gt": {
                "r_bar_mean": float(np.mean(gt_r_bars)),
                "r_bar_std": float(np.std(gt_r_bars)),
                "epl_mean": float(np.mean(gt_epls)),
                "epl_std": float(np.std(gt_epls)),
                "n_examples": len(examples),
                "n_pilot_gt": 30,
            },
            "pilot_estimates": pilot_results,
        }

    with open(os.path.join(output_dir, "proxy_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    return validation


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLOX v2: Full NeurIPS Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel (MUST be 2)")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["pilot", "topology", "strategies", "proxy", "all"])
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated benchmark list (default: auto from pilot)")
    parser.add_argument("--strategies", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="11,23,37")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output", type=str, default=OUTPUT_BASE)
    parser.add_argument("--quant", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)

    seeds = [int(s) for s in args.seeds.split(",")]
    strategy_names = (
        [s.strip() for s in args.strategies.split(",")]
        if args.strategies
        else CORE_STRATEGIES
    )

    log.info("CLOX v2 Full Experiment")
    log.info(f"  Model: {args.model}")
    log.info(f"  TP: {args.tp}")
    log.info(f"  Phase: {args.phase}")
    log.info(f"  Seeds: {seeds}")
    log.info(f"  Output: {args.output}")

    model_tag = args.model.split("/")[-1]
    model_dir = os.path.join(args.output, model_tag)
    os.makedirs(model_dir, exist_ok=True)

    engine = init_engine(args.model, tp=args.tp, quant=args.quant)

    # Determine benchmarks
    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    elif args.phase == "pilot":
        benchmarks = ["gsm8k", "math", "strategyqa", "arc_challenge"]
    else:
        benchmarks = ["gsm8k", "math", "strategyqa", "arc_challenge"]

    # Phase 0: Pilot
    if args.phase in ("pilot", "all"):
        pilot_dir = os.path.join(model_dir, "pilot")
        _, recommended = run_pilot(engine, pilot_dir, n_pilot=50)
        if args.phase == "all" and recommended:
            benchmarks = recommended
        log.info(f"Using benchmarks: {benchmarks}")

    # Phase 1: Topology
    if args.phase in ("topology", "all"):
        run_topology(engine, benchmarks, model_dir, n_examples=200, n_pilot=5)

    # Phase 2: Strategies
    if args.phase in ("strategies", "all"):
        run_strategies(
            engine, benchmarks, strategy_names, seeds, model_dir,
            max_examples=args.max_examples,
        )

    # Phase 3: Proxy validation
    if args.phase in ("proxy", "all"):
        run_proxy_validation(engine, benchmarks[:2], model_dir,
                            n_gt=50, n_pilot_list=[3, 5, 8])

    log.info("\n=== EXPERIMENT COMPLETE ===")
    log.info(f"Results in: {model_dir}")

    # Print final summary
    log.info("\n=== FINAL SUMMARY ===")
    for bname in benchmarks:
        strat_file = os.path.join(model_dir, bname, "strategies.json")
        if os.path.exists(strat_file):
            with open(strat_file) as f:
                d = json.load(f)
            log.info(f"\n{bname}:")
            for sname in d.get("strategies", []):
                a = d.get("aggregate", {}).get(sname, {})
                log.info(f"  {sname}: {a.get('mean_accuracy', 0):.4f} "
                         f"± {a.get('std_accuracy', 0):.4f}")


if __name__ == "__main__":
    main()
