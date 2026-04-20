#!/usr/bin/env python3
"""Cross-Strategy Pilot: Validate Idea A's core premise.

Run 8 base strategies on 50 GSM8K problems (difficulty-balanced,
oversampling hard) to measure pairwise error correlation and determine
whether cross-strategy voting can beat single-strategy SC.

Decision rule (applied by analyze_pilot.py):
  |phi| < 0.3 between >= 3 strategy pairs  -->  PROCEED with Idea A
  |phi| > 0.7 for all pairs                -->  ABANDON, consider Idea B

Usage:
    python run_pilot.py                          # auto-detect GPUs and TP
    python run_pilot.py --n_problems 50          # custom problem count
    python run_pilot.py --tp 2                   # force TP=2
    python run_pilot.py --skip_topology          # faster, no topology estimation
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# Apply offline HF patch BEFORE any vllm/transformers imports
if os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "True"):
    import hf_offline_patch  # noqa: F401

from benchmarks import FEW_SHOT_PROMPTS, load_benchmark
from engine import VLLMEngine, auto_tp
from evaluation import check_answer
from strategies_v2 import build_strategy
from topology_v2 import batch_estimate_topology

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("pilot")

# Pre-create output dir so shell redirection (nohup ... > results/pilot/x.log)
# works before main() runs
_default_output = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "pilot",
)
os.makedirs(_default_output, exist_ok=True)

# 8 base strategies (exclude clox_adaptive meta-strategy)
PILOT_STRATEGIES = [
    "standard_cot",
    "self_consistency",
    "compute_matched_sc",
    "targeted_repair",
    "random_repair",
    "backward_cloze",
    "full_regeneration",
    "hierarchical_repair",
]

# Strategy-specific kwargs
STRATEGY_KWARGS: dict[str, dict] = {
    "self_consistency": {"k": 8},
}


def select_problems(n: int = 50, seed: int = 42) -> list:
    """Select n GSM8K problems, oversampling hard for signal.

    Target: 40% easy, 36% medium, 24% hard (vs natural 58/37/5).
    Oversampling hard gives enough signal to detect topology effects.
    """
    all_examples = load_benchmark("gsm8k")
    by_diff: dict[str, list] = {}
    for ex in all_examples:
        by_diff.setdefault(ex.difficulty, []).append(ex)

    rng = np.random.default_rng(seed)
    targets = {"easy": max(1, int(n * 0.40)),
               "medium": max(1, int(n * 0.36)),
               "hard": max(1, n - int(n * 0.40) - int(n * 0.36))}

    selected = []
    for diff, target in targets.items():
        pool = by_diff.get(diff, [])
        k = min(target, len(pool))
        if k > 0:
            idx = rng.choice(len(pool), size=k, replace=False)
            selected.extend(pool[i] for i in sorted(idx))

    rng.shuffle(selected)
    return selected[:n]


def _load_ckpt(path: str) -> dict:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_ckpt(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, path)


def run_strategies(engine: VLLMEngine, examples: list, output_dir: str, few_shot: str,
                   max_tokens: int = 2048) -> dict:
    """Run all 8 strategies on all examples with checkpoint/resume."""
    ckpt_path = os.path.join(output_dir, ".pilot_ckpt.json")
    results = _load_ckpt(ckpt_path)
    if results:
        total_done = sum(len(v) for v in results.values())
        log.info("Resumed from checkpoint: %d entries", total_done)

    for sname in PILOT_STRATEGIES:
        if sname not in results:
            results[sname] = {}

        kwargs = STRATEGY_KWARGS.get(sname, {})
        strategy = build_strategy(sname, **kwargs)

        remaining = [ex for ex in examples if ex.example_id not in results[sname]]
        if not remaining:
            acc = sum(1 for v in results[sname].values() if v.get("correct")) / max(len(results[sname]), 1)
            log.info("  %s: already done (%d examples, acc=%.3f)", sname, len(results[sname]), acc)
            continue

        log.info("  %s: %d done, %d remaining", sname, len(results[sname]), len(remaining))

        for i, ex in enumerate(remaining):
            t0 = time.perf_counter()
            try:
                result = strategy.run(engine, ex.question, max_tokens=max_tokens, few_shot=few_shot)
                elapsed = time.perf_counter() - t0
                correct = check_answer(result.prediction, ex.answer, ex.answer_type)

                results[sname][ex.example_id] = {
                    "example_id": ex.example_id,
                    "difficulty": ex.difficulty,
                    "prediction": result.prediction,
                    "ground_truth": ex.answer,
                    "correct": correct,
                    "confidence": result.confidence,
                    "total_tokens": result.total_tokens,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "reasoning_trace": result.reasoning_trace[:2000],
                    "step_metadata": result.step_metadata,
                    "elapsed_s": elapsed,
                }
            except Exception as e:
                log.warning("Error %s/%s: %s", sname, ex.example_id, e)
                results[sname][ex.example_id] = {
                    "example_id": ex.example_id,
                    "difficulty": ex.difficulty,
                    "correct": False,
                    "error": str(e),
                }

            if (i + 1) % 10 == 0:
                done = len(results[sname])
                acc = sum(1 for v in results[sname].values() if v.get("correct")) / done
                log.info("    %s: %d/%d acc=%.3f", sname, done, len(examples), acc)
                _save_ckpt(ckpt_path, results)

        _save_ckpt(ckpt_path, results)
        acc = sum(1 for v in results[sname].values() if v.get("correct")) / max(len(results[sname]), 1)
        tokens = [v.get("total_tokens", 0) for v in results[sname].values() if v.get("total_tokens", 0) > 0]
        log.info("  %s done: acc=%.3f mean_tokens=%.0f", sname, acc, np.mean(tokens) if tokens else 0)

    return results


def run_topology_estimation(engine: VLLMEngine, examples: list, few_shot: str) -> dict:
    """Estimate (r_bar, epl) topology for each problem."""
    log.info("Estimating topology for %d problems...", len(examples))
    questions = [ex.question for ex in examples]

    # Small batches to avoid OOM
    batch_size = 10
    all_profiles = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        log.info("  Topology batch %d-%d / %d", start, start + len(batch), len(questions))
        profiles = batch_estimate_topology(
            engine, batch, few_shot=few_shot, n_pilot=8, max_tokens=2048,
        )
        all_profiles.extend(profiles)

    topology = {}
    for ex, prof in zip(examples, all_profiles):
        topology[ex.example_id] = {
            "r_bar": prof.r_bar,
            "epl": prof.epl,
            "n_steps": prof.n_steps,
            "strategy": prof.strategy,
        }
    log.info("Topology estimation complete")
    return topology


def main():
    parser = argparse.ArgumentParser(description="Cross-Strategy Pilot Experiment")
    parser.add_argument("--n_problems", type=int, default=50)
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: results/pilot)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--tp", type=int, default=0,
                        help="Tensor parallel size (0 = auto-detect via auto_tp)")
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--quant", type=str, default=None,
                        help="Quantization method: awq, gptq, or None")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max generation tokens per call (512 too short for Qwen3.5)")
    parser.add_argument("--skip_topology", action="store_true",
                        help="Skip topology estimation (faster)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing checkpoint and start fresh")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index (0-based) for data parallelism")
    parser.add_argument("--n_shards", type=int, default=1,
                        help="Total number of shards (splits problems evenly)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "pilot",
        )
    os.makedirs(args.output, exist_ok=True)

    # Clean old checkpoint if requested
    if args.clean:
        ckpt = os.path.join(args.output, ".pilot_ckpt.json")
        if os.path.isfile(ckpt):
            os.unlink(ckpt)
            print(f"Deleted old checkpoint: {ckpt}")

    # File logger
    fh = logging.FileHandler(os.path.join(args.output, "pilot.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    log.info("=== Cross-Strategy Pilot Experiment ===")
    log.info("Model: %s  N=%d  seed=%d", args.model, args.n_problems, args.seed)

    # Select problems (full 50, then take shard)
    all_examples = select_problems(args.n_problems, seed=args.seed)
    if args.n_shards > 1:
        examples = all_examples[args.shard_id::args.n_shards]
        log.info("Shard %d/%d: %d/%d problems",
                 args.shard_id, args.n_shards, len(examples), len(all_examples))
    else:
        examples = all_examples
    diff_counts: dict[str, int] = {}
    for ex in examples:
        diff_counts[ex.difficulty] = diff_counts.get(ex.difficulty, 0) + 1
    log.info("Selected %d problems: %s", len(examples), diff_counts)

    # Save problem list for reproducibility
    problem_list = [{
        "id": ex.example_id,
        "difficulty": ex.difficulty,
        "question": ex.question[:200],
        "answer": ex.answer,
    } for ex in examples]
    with open(os.path.join(args.output, "problems.json"), "w") as f:
        json.dump(problem_list, f, indent=2)

    # Init engine (auto-detect TP if not specified)
    tp = args.tp if args.tp > 0 else auto_tp(args.model)
    log.info("Loading model: %s (tp=%d%s)", args.model, tp,
             ", auto-detected" if args.tp == 0 else "")
    engine = VLLMEngine(
        model_name=args.model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        quantization=args.quant,
        seed=args.seed,
    )
    log.info("Model loaded")

    few_shot = FEW_SHOT_PROMPTS.get("gsm8k", "")

    # Phase 1: Run all strategies
    log.info("--- Phase 1: Running %d strategies ---", len(PILOT_STRATEGIES))
    t_start = time.perf_counter()
    strategy_results = run_strategies(engine, examples, args.output, few_shot,
                                      max_tokens=args.max_tokens)
    t_strat = time.perf_counter() - t_start
    log.info("Strategies complete in %.1f min", t_strat / 60)

    # Phase 2: Topology estimation
    topology = {}
    if not args.skip_topology:
        log.info("--- Phase 2: Topology estimation ---")
        t0 = time.perf_counter()
        topology = run_topology_estimation(engine, examples, few_shot)
        log.info("Topology complete in %.1f min", (time.perf_counter() - t0) / 60)

    # Save final results
    output = {
        "config": {
            "model": args.model,
            "n_problems": len(examples),
            "strategies": PILOT_STRATEGIES,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "strategy_time_min": round(t_strat / 60, 1),
        },
        "problems": problem_list,
        "topology": topology,
        "strategy_results": strategy_results,
    }

    out_path = os.path.join(args.output, "pilot_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Clean checkpoint
    ckpt = os.path.join(args.output, ".pilot_ckpt.json")
    if os.path.isfile(ckpt):
        os.unlink(ckpt)

    # Summary
    log.info("\n=== PILOT SUMMARY ===")
    for sname in PILOT_STRATEGIES:
        sres = strategy_results.get(sname, {})
        n = len(sres)
        acc = sum(1 for v in sres.values() if v.get("correct")) / max(n, 1)
        tokens = [v.get("total_tokens", 0) for v in sres.values()
                  if v.get("total_tokens", 0) > 0]
        log.info("  %-25s acc=%.3f  tokens=%5.0f  n=%d",
                 sname, acc, np.mean(tokens) if tokens else 0, n)

    log.info("\nResults: %s", out_path)
    log.info("Next:    python analyze_pilot.py %s", args.output)


if __name__ == "__main__":
    main()
