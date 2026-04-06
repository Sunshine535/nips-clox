#!/usr/bin/env python3
"""CLOX v2: Topology-Aware Inference-Time Strategy Selection.

Main experiment runner using vLLM for high-throughput inference.

Usage:
    python run_clox.py --model Qwen/Qwen2.5-72B-Instruct --tp 2 \
        --benchmarks gsm8k,math --phase topology
    python run_clox.py --model Qwen/Qwen2.5-72B-Instruct --tp 2 \
        --benchmarks gsm8k,math --phase strategies --seeds 11,23,37,47,59
    python run_clox.py --model Qwen/Qwen2.5-72B-Instruct --tp 2 \
        --benchmarks gsm8k,math --phase adaptive
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from benchmarks import BenchmarkExample, FEW_SHOT_PROMPTS, load_benchmark
from engine import VLLMEngine, extract_answer
from evaluation import (
    bonferroni_correction,
    check_answer,
    cohens_d,
    compute_token_efficiency,
    mcnemar_test,
    paired_bootstrap_ci,
)
from strategies_v2 import STRATEGY_REGISTRY, StrategyResult, build_strategy
from topology_v2 import TopologyProfile, batch_estimate_topology, estimate_topology

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("clox")


# ── Phase 1: Topology Characterization ──────────────────────────────

def run_topology_phase(
    engine: VLLMEngine,
    benchmark_name: str,
    examples: list[BenchmarkExample],
    output_dir: str,
    n_pilot: int = 8,
    max_tokens: int = 512,
):
    """Characterize reasoning topology across a benchmark.

    For each example, estimate (r̄, ℓ) and record per-step metrics.
    This builds the phase diagram showing where different strategies
    are expected to be optimal.
    """
    log.info("=== Phase 1: Topology Characterization ===")
    log.info("Benchmark: %s, %d examples, %d pilot traces each",
             benchmark_name, len(examples), n_pilot)

    few_shot = FEW_SHOT_PROMPTS.get(benchmark_name.split("_")[0], "")
    results = []

    # Batch process for efficiency
    batch_size = 32
    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start:batch_start + batch_size]
        questions = [ex.question for ex in batch]

        log.info("Processing batch %d-%d / %d",
                 batch_start, batch_start + len(batch), len(examples))

        profiles = batch_estimate_topology(
            engine, questions, few_shot=few_shot,
            n_pilot=n_pilot, max_tokens=max_tokens,
        )

        for ex, profile in zip(batch, profiles):
            results.append({
                "example_id": ex.example_id,
                "benchmark": ex.benchmark,
                "difficulty": ex.difficulty,
                "r_bar": profile.r_bar,
                "epl": profile.epl,
                "n_steps": profile.n_steps,
                "recommended_strategy": profile.strategy,
                "question_preview": ex.question[:100],
            })

    # Summary statistics
    r_bars = [r["r_bar"] for r in results]
    epls = [r["epl"] for r in results]
    strategies = [r["recommended_strategy"] for r in results]

    summary = {
        "benchmark": benchmark_name,
        "n_examples": len(results),
        "r_bar_mean": float(np.mean(r_bars)),
        "r_bar_std": float(np.std(r_bars)),
        "r_bar_median": float(np.median(r_bars)),
        "epl_mean": float(np.mean(epls)),
        "epl_std": float(np.std(epls)),
        "epl_median": float(np.median(epls)),
        "strategy_distribution": {
            s: strategies.count(s) for s in set(strategies)
        },
        "per_difficulty": {},
    }

    for diff in set(r["difficulty"] for r in results):
        subset = [r for r in results if r["difficulty"] == diff]
        summary["per_difficulty"][diff] = {
            "n": len(subset),
            "r_bar_mean": float(np.mean([r["r_bar"] for r in subset])),
            "epl_mean": float(np.mean([r["epl"] for r in subset])),
        }

    log.info("Topology Summary:")
    log.info("  r̄ = %.3f ± %.3f", summary["r_bar_mean"], summary["r_bar_std"])
    log.info("  ℓ = %.2f ± %.2f", summary["epl_mean"], summary["epl_std"])
    log.info("  Strategy distribution: %s", summary["strategy_distribution"])

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{benchmark_name}_topology.json"), "w") as f:
        json.dump({"summary": summary, "per_example": results}, f, indent=2)

    log.info("Saved to %s/%s_topology.json", output_dir, benchmark_name)
    return summary


# ── Phase 2: Strategy Comparison ────────────────────────────────────

def run_strategy_phase(
    engine: VLLMEngine,
    benchmark_name: str,
    examples: list[BenchmarkExample],
    strategy_names: list[str],
    seeds: list[int],
    output_dir: str,
    max_tokens: int = 512,
):
    """Run all strategies on all examples with multiple seeds.

    Records per-example correctness, token counts, and timing
    for statistical comparison.
    """
    log.info("=== Phase 2: Strategy Comparison ===")
    log.info("Benchmark: %s, %d examples, strategies: %s, seeds: %s",
             benchmark_name, len(examples), strategy_names, seeds)

    few_shot = FEW_SHOT_PROMPTS.get(benchmark_name.split("_")[0], "")
    all_results: dict[str, dict[int, list[dict]]] = {}

    for strategy_name in strategy_names:
        all_results[strategy_name] = {}
        strategy = build_strategy(strategy_name)

        for seed in seeds:
            ckpt_path = os.path.join(output_dir, f".ckpt_{benchmark_name}_{strategy_name}_s{seed}.json")
            cached = _load_ckpt(ckpt_path)
            done_ids = {r["example_id"] for r in cached}
            remaining = [ex for ex in examples if ex.example_id not in done_ids]

            if cached:
                log.info("Resuming %s/seed=%d: %d done, %d remaining",
                         strategy_name, seed, len(cached), len(remaining))

            # Set seed in vLLM engine
            engine.llm.llm_engine.model_config.seed = seed

            results = list(cached)
            for i, ex in enumerate(remaining):
                t0 = time.perf_counter()
                try:
                    result = strategy.run(
                        engine, ex.question,
                        max_tokens=max_tokens, few_shot=few_shot,
                    )
                    elapsed = (time.perf_counter() - t0) * 1000
                    is_correct = check_answer(result.prediction, ex.answer, ex.answer_type)

                    results.append({
                        "example_id": ex.example_id,
                        "benchmark": ex.benchmark,
                        "difficulty": ex.difficulty,
                        "prediction": result.prediction,
                        "ground_truth": ex.answer,
                        "correct": is_correct,
                        "confidence": result.confidence,
                        "total_tokens": result.total_tokens,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "strategy": result.strategy_name,
                        "elapsed_ms": elapsed,
                        "seed": seed,
                        "reasoning_trace": result.reasoning_trace[:500],
                        "topology": {
                            "r_bar": result.topology.r_bar if result.topology else None,
                            "epl": result.topology.epl if result.topology else None,
                            "selected": result.topology.strategy if result.topology else None,
                        },
                    })
                except Exception as exc:
                    log.warning("Error on %s/%s: %s", strategy_name, ex.example_id, exc)
                    results.append({
                        "example_id": ex.example_id,
                        "correct": False,
                        "total_tokens": 0,
                        "error": str(exc),
                        "seed": seed,
                    })

                if (i + 1) % 50 == 0:
                    acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
                    log.info("  %s seed=%d: %d/%d done, acc=%.4f",
                             strategy_name, seed, len(results), len(examples), acc)
                    _save_ckpt(ckpt_path, results)

            _save_ckpt(ckpt_path, results)
            all_results[strategy_name][seed] = results

            acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
            tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens", 0) > 0]
            log.info("  %s seed=%d: accuracy=%.4f, mean_tokens=%.0f, n=%d",
                     strategy_name, seed, acc, np.mean(tokens) if tokens else 0, len(results))

    # Compute aggregate metrics
    aggregate = _compute_aggregate(all_results, strategy_names, seeds)

    output = {
        "benchmark": benchmark_name,
        "n_examples": len(examples),
        "strategies": strategy_names,
        "seeds": seeds,
        "aggregate": aggregate,
        "per_strategy_results": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{benchmark_name}_strategies.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved to %s", out_path)

    # Clean checkpoints
    for p in Path(output_dir).glob(f".ckpt_{benchmark_name}_*.json"):
        p.unlink(missing_ok=True)

    return aggregate


# ── Phase 3: Adaptive Routing Evaluation ─────────────────────────────

def run_adaptive_phase(
    engine: VLLMEngine,
    benchmark_name: str,
    examples: list[BenchmarkExample],
    seeds: list[int],
    output_dir: str,
    max_tokens: int = 512,
):
    """Evaluate CLOX-Adaptive: topology-guided strategy selection.

    For each example:
    1. Estimate topology (r̄, ℓ)
    2. Select strategy based on topology
    3. Execute and record results + token accounting

    Compare against fixed strategies and oracle selection.
    """
    log.info("=== Phase 3: Adaptive Routing ===")
    few_shot = FEW_SHOT_PROMPTS.get(benchmark_name.split("_")[0], "")
    adaptive = build_strategy("clox_adaptive")

    all_results: dict[int, list[dict]] = {}
    routing_stats = {"strategies_selected": {}}

    for seed in seeds:
        engine.llm.llm_engine.model_config.seed = seed
        results = []

        for i, ex in enumerate(examples):
            t0 = time.perf_counter()
            try:
                result = adaptive.run(engine, ex.question, max_tokens=max_tokens, few_shot=few_shot)
                elapsed = (time.perf_counter() - t0) * 1000
                is_correct = check_answer(result.prediction, ex.answer, ex.answer_type)

                selected = result.strategy_name  # "clox_adaptive(X)"

                results.append({
                    "example_id": ex.example_id,
                    "correct": is_correct,
                    "total_tokens": result.total_tokens,
                    "prediction": result.prediction,
                    "ground_truth": ex.answer,
                    "selected_strategy": selected,
                    "topology": {
                        "r_bar": result.topology.r_bar if result.topology else None,
                        "epl": result.topology.epl if result.topology else None,
                    },
                    "elapsed_ms": elapsed,
                    "seed": seed,
                })

                sel = selected.split("(")[-1].rstrip(")") if "(" in selected else selected
                routing_stats["strategies_selected"][sel] = \
                    routing_stats["strategies_selected"].get(sel, 0) + 1

            except Exception as exc:
                log.warning("Error on adaptive/%s: %s", ex.example_id, exc)
                results.append({
                    "example_id": ex.example_id, "correct": False,
                    "total_tokens": 0, "error": str(exc), "seed": seed,
                })

            if (i + 1) % 50 == 0:
                acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
                log.info("  adaptive seed=%d: %d/%d, acc=%.4f", seed, i + 1, len(examples), acc)

        all_results[seed] = results
        acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
        tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens", 0) > 0]
        log.info("  adaptive seed=%d: accuracy=%.4f, mean_tokens=%.0f", seed, acc,
                 np.mean(tokens) if tokens else 0)

    output = {
        "benchmark": benchmark_name,
        "n_examples": len(examples),
        "seeds": seeds,
        "routing_stats": routing_stats,
        "per_seed_results": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{benchmark_name}_adaptive.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved to %s", out_path)
    return output


# ── Helpers ──────────────────────────────────────────────────────────

def _load_ckpt(path: str) -> list[dict]:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return []

def _save_ckpt(path: str, data: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, path)


def _compute_aggregate(
    all_results: dict[str, dict[int, list[dict]]],
    strategy_names: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    aggregate = {}

    for sname in strategy_names:
        seed_accs = []
        all_correct = []
        all_tokens = []
        for seed in seeds:
            results = all_results.get(sname, {}).get(seed, [])
            correct = [r.get("correct", False) for r in results]
            acc = sum(correct) / max(len(correct), 1)
            seed_accs.append(acc)
            all_correct.extend(correct)
            all_tokens.extend(r.get("total_tokens", 0) for r in results)

        aggregate[sname] = {
            "mean_accuracy": float(np.mean(seed_accs)) if seed_accs else 0.0,
            "std_accuracy": float(np.std(seed_accs)) if len(seed_accs) > 1 else 0.0,
            "per_seed_accuracy": {s: a for s, a in zip(seeds, seed_accs)},
            "total_examples": len(all_correct),
            "total_correct": sum(all_correct),
            "mean_tokens": float(np.mean(all_tokens)) if all_tokens else 0.0,
            "token_efficiency": compute_token_efficiency(
                [1.0 if c else 0.0 for c in all_correct], all_tokens,
            ) if all_tokens else {},
        }

    # Pairwise tests vs baseline
    baseline = "standard_cot"
    if baseline in aggregate:
        pairwise = {}
        for sname in strategy_names:
            if sname == baseline:
                continue
            s_correct, b_correct = [], []
            for seed in seeds:
                s_correct.extend(r.get("correct", False)
                                 for r in all_results.get(sname, {}).get(seed, []))
                b_correct.extend(r.get("correct", False)
                                 for r in all_results.get(baseline, {}).get(seed, []))
            boot = paired_bootstrap_ci(s_correct, b_correct)
            mcn = mcnemar_test(s_correct, b_correct)
            cd = cohens_d(
                [1.0 if c else 0.0 for c in s_correct],
                [1.0 if c else 0.0 for c in b_correct],
            )
            pairwise[f"{sname}_vs_{baseline}"] = {
                "bootstrap": boot, "mcnemar": mcn, "cohens_d": cd,
            }
        aggregate["pairwise_vs_baseline"] = pairwise

    return aggregate


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLOX v2: Topology-Aware Strategy Selection")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--benchmarks", type=str, default="gsm8k")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["topology", "strategies", "adaptive", "all"])
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="11,23,37,47,59")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results/v2")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method: awq, gptq, None")
    parser.add_argument("--gpu_mem", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--n_pilot", type=int, default=8,
                        help="Number of pilot traces for topology estimation")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.strategies == "all":
        strategy_names = list(STRATEGY_REGISTRY.keys())
    else:
        strategy_names = [s.strip() for s in args.strategies.split(",")]

    log.info("CLOX v2 Configuration:")
    log.info("  Model: %s", args.model)
    log.info("  Tensor parallel: %d", args.tp)
    log.info("  Benchmarks: %s", benchmarks)
    log.info("  Phase: %s", args.phase)
    log.info("  Strategies: %s", strategy_names)
    log.info("  Seeds: %s", seeds)
    log.info("  Quantization: %s", args.quantization)

    model_dir = os.path.join(args.output_dir, args.model.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    log.info("Loading model via vLLM: %s", args.model)
    engine = VLLMEngine(
        model_name=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
    )
    log.info("Model loaded successfully")

    for bname in benchmarks:
        log.info("\n{'='*60}")
        log.info("Benchmark: %s", bname)
        log.info("{'='*60}")

        examples = load_benchmark(bname, max_examples=args.max_examples)
        log.info("Loaded %d examples", len(examples))

        bm_dir = os.path.join(model_dir, bname)

        if args.phase in ("topology", "all"):
            run_topology_phase(
                engine, bname, examples, bm_dir,
                n_pilot=args.n_pilot, max_tokens=args.max_tokens,
            )

        if args.phase in ("strategies", "all"):
            run_strategy_phase(
                engine, bname, examples, strategy_names, seeds, bm_dir,
                max_tokens=args.max_tokens,
            )

        if args.phase in ("adaptive", "all"):
            run_adaptive_phase(
                engine, bname, examples, seeds, bm_dir,
                max_tokens=args.max_tokens,
            )

    log.info("\n=== DONE ===")
    log.info("Results in: %s", model_dir)


if __name__ == "__main__":
    main()
