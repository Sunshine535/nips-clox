from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from benchmarks import FEW_SHOT_PROMPTS, BenchmarkExample, load_benchmark
from evaluation import (
    bonferroni_correction,
    check_answer,
    cohens_d,
    compute_task_topology_metrics,
    compute_token_efficiency,
    exact_match_accuracy,
    mcnemar_test,
    paired_bootstrap_ci,
    per_example_win_loss_matrix,
)
from strategies import STRATEGY_REGISTRY, StrategyResult, build_strategy
from topology import TopologyEstimator, compute_theoretical_error_bound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("clox")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, quantize: bool = False, device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }

    if quantize:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    if device:
        model_kwargs["device_map"] = {"": device}
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    return model, tokenizer


# ── Checkpoint helpers ────────────────────────────────────────────────

def _ckpt_path(output_dir: str, benchmark: str, strategy: str, seed: int) -> str:
    return os.path.join(output_dir, f".ckpt_{benchmark}_{strategy}_s{seed}.json")


def _load_checkpoint(path: str) -> list[dict[str, Any]]:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_checkpoint(path: str, results: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, default=str)
    os.replace(tmp, path)


def _clear_checkpoints(output_dir: str, benchmark: str) -> None:
    for p in Path(output_dir).glob(f".ckpt_{benchmark}_*.json"):
        p.unlink(missing_ok=True)


def _gpu_ckpt_path(output_dir: str, benchmark: str, strategy: str, seed: int, rank: int) -> str:
    return os.path.join(output_dir, f".ckpt_{benchmark}_{strategy}_s{seed}_gpu{rank}.json")


# ── Single-example runner ─────────────────────────────────────────────

def run_single_example(
    strategy,
    model,
    tokenizer,
    example: BenchmarkExample,
    max_new_tokens: int,
    few_shot_prompt: str,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    result: StrategyResult = strategy.run(
        model=model,
        tokenizer=tokenizer,
        question=example.question,
        max_new_tokens=max_new_tokens,
        few_shot_prompt=few_shot_prompt,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    is_correct = check_answer(result.prediction, example.answer, example.answer_type)

    return {
        "example_id": example.example_id,
        "benchmark": example.benchmark,
        "difficulty": example.difficulty,
        "question": example.question[:200],
        "ground_truth": example.answer,
        "prediction": result.prediction,
        "correct": is_correct,
        "confidence": result.confidence,
        "total_tokens": result.total_tokens,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "strategy": result.strategy_name,
        "elapsed_ms": elapsed_ms,
        "logprobs": result.logprobs,
        "step_metadata": result.step_metadata,
        "reasoning_trace": result.reasoning_trace,
        "n_logprobs": len(result.logprobs),
        "mean_logprob": float(np.mean(result.logprobs)) if result.logprobs else 0.0,
        "topology": {
            "epl": result.topology.epl if result.topology else None,
            "recoverability": result.topology.local_recoverability if result.topology else None,
            "recommendation": result.topology.strategy_recommendation if result.topology else None,
        },
    }


# ── Multi-GPU worker ──────────────────────────────────────────────────

def _gpu_worker(
    rank: int,
    world_size: int,
    model_name: str,
    quantize: bool,
    benchmark_name: str,
    examples: list[dict],
    strategy_name: str,
    seed: int,
    max_new_tokens: int,
    few_shot_prompt: str,
    output_dir: str,
    return_dict: dict,
):
    """Run a shard of examples on GPU `rank`."""
    device = f"cuda:{rank}"
    set_seed(seed + rank)

    model, tokenizer = load_model(model_name, quantize=quantize, device=device)
    strategy = build_strategy(strategy_name)

    my_examples = [
        BenchmarkExample(**ex) for i, ex in enumerate(examples) if i % world_size == rank
    ]

    gpu_ckpt = _gpu_ckpt_path(output_dir, benchmark_name, strategy_name, seed, rank)

    results = _load_checkpoint(gpu_ckpt)
    done_ids = {r["example_id"] for r in results}
    my_examples = [ex for ex in my_examples if ex.example_id not in done_ids]

    for i, example in enumerate(my_examples):
        try:
            result = run_single_example(
                strategy, model, tokenizer, example, max_new_tokens, few_shot_prompt,
            )
            result["seed"] = seed
            results.append(result)
        except Exception as exc:
            logger.warning("[GPU %d] Error on %s: %s", rank, example.example_id, exc)
            results.append({
                "example_id": example.example_id,
                "correct": False,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(exc),
                "seed": seed,
            })
        if (i + 1) % 20 == 0:
            acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
            logger.info("[GPU %d] %s/%s seed=%d: %d/%d done, acc=%.4f",
                        rank, strategy_name, benchmark_name, seed, i + 1, len(my_examples), acc)
            _save_checkpoint(gpu_ckpt, results)

    _save_checkpoint(gpu_ckpt, results)
    return_dict[rank] = results
    del model
    torch.cuda.empty_cache()


# ── Main experiment loop ──────────────────────────────────────────────

def run_experiment(
    model,
    tokenizer,
    model_name: str,
    benchmark_name: str,
    examples: list[BenchmarkExample],
    strategy_names: list[str],
    seeds: list[int],
    max_new_tokens: int,
    output_dir: str,
    n_gpus: int = 1,
    quantize: bool = False,
) -> dict[str, Any]:
    few_shot = FEW_SHOT_PROMPTS.get(benchmark_name.split("_")[0], "")

    all_results: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for strategy_name in strategy_names:
        all_results[strategy_name] = {}
        for seed in seeds:
            ckpt = _ckpt_path(output_dir, benchmark_name, strategy_name, seed)
            cached = _load_checkpoint(ckpt)
            done_ids = {r["example_id"] for r in cached}
            remaining = [ex for ex in examples if ex.example_id not in done_ids]

            if cached:
                logger.info(
                    "Resuming %s/%s seed=%d: %d done, %d remaining",
                    strategy_name, benchmark_name, seed, len(cached), len(remaining),
                )

            if remaining:
                if n_gpus > 1 and model is None:
                    new_results = _run_multi_gpu(
                        n_gpus, model_name, quantize, benchmark_name, remaining,
                        strategy_name, seed, max_new_tokens, few_shot, output_dir,
                    )
                else:
                    set_seed(seed)
                    strategy = build_strategy(strategy_name)
                    new_results = []
                    for i, example in enumerate(remaining):
                        try:
                            result = run_single_example(
                                strategy, model, tokenizer, example, max_new_tokens, few_shot,
                            )
                            result["seed"] = seed
                            new_results.append(result)
                        except Exception as exc:
                            logger.warning("Error on %s: %s", example.example_id, exc)
                            new_results.append({
                                "example_id": example.example_id,
                                "correct": False,
                                "total_tokens": 0,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "error": str(exc),
                                "seed": seed,
                            })
                        if (i + 1) % 50 == 0:
                            acc = sum(r.get("correct", False) for r in (cached + new_results)) / max(len(cached) + len(new_results), 1)
                            logger.info("  %s/%s seed=%d: %d/%d, acc=%.4f",
                                        strategy_name, benchmark_name, seed,
                                        len(cached) + i + 1, len(examples), acc)
                            _save_checkpoint(ckpt, cached + new_results)

                cached.extend(new_results)
                _save_checkpoint(ckpt, cached)

            id_order = {ex.example_id: idx for idx, ex in enumerate(examples)}
            cached.sort(key=lambda r: id_order.get(r.get("example_id", ""), len(examples)))
            all_results[strategy_name][seed] = cached

            acc = sum(r.get("correct", False) for r in cached) / max(len(cached), 1)
            tokens = [r.get("total_tokens", 0) for r in cached if "total_tokens" in r]
            logger.info(
                "  %s seed=%d: accuracy=%.4f, mean_tokens=%.0f, n=%d",
                strategy_name, seed, acc, np.mean(tokens) if tokens else 0, len(cached),
            )

    aggregate = compute_aggregate_metrics(all_results, strategy_names, seeds)

    experiment_record = {
        "benchmark": benchmark_name,
        "n_examples": len(examples),
        "strategy_names": strategy_names,
        "seeds": seeds,
        "max_new_tokens": max_new_tokens,
        "per_strategy_results": all_results,
        "aggregate": aggregate,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{benchmark_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(experiment_record, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    _clear_checkpoints(output_dir, benchmark_name)
    return experiment_record


def _run_multi_gpu(
    n_gpus, model_name, quantize, benchmark_name, examples,
    strategy_name, seed, max_new_tokens, few_shot, output_dir,
) -> list[dict[str, Any]]:
    """Data-parallel inference across GPUs using mp.spawn."""
    os.makedirs(output_dir, exist_ok=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    serialized = [
        {"example_id": ex.example_id, "question": ex.question,
         "answer": ex.answer, "answer_type": ex.answer_type,
         "benchmark": ex.benchmark, "difficulty": ex.difficulty,
         "metadata": ex.metadata}
        for ex in examples
    ]

    try:
        mp.spawn(
            _gpu_worker,
            args=(
                n_gpus, model_name, quantize, benchmark_name, serialized,
                strategy_name, seed, max_new_tokens, few_shot, output_dir, return_dict,
            ),
            nprocs=n_gpus,
            join=True,
        )
    except Exception as exc:
        logger.warning("Multi-GPU spawn error: %s. Recovering partial results.", exc)

    combined = []
    for rank in range(n_gpus):
        gpu_ckpt = _gpu_ckpt_path(output_dir, benchmark_name, strategy_name, seed, rank)
        if rank in return_dict:
            combined.extend(return_dict[rank])
        else:
            partial = _load_checkpoint(gpu_ckpt)
            if partial:
                logger.info("Recovered %d results from GPU %d checkpoint", len(partial), rank)
                combined.extend(partial)
        Path(gpu_ckpt).unlink(missing_ok=True)

    return combined


def compute_aggregate_metrics(
    all_results: dict[str, dict[int, list[dict[str, Any]]]],
    strategy_names: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}

    for strategy_name in strategy_names:
        seed_accuracies = []
        all_correct = []
        all_tokens = []

        for seed in seeds:
            results = all_results.get(strategy_name, {}).get(seed, [])
            correct_list = [r.get("correct", False) for r in results]
            acc = sum(correct_list) / max(len(correct_list), 1)
            seed_accuracies.append(acc)
            all_correct.extend(correct_list)
            all_tokens.extend([r.get("total_tokens", 0) for r in results])

        mean_acc = float(np.mean(seed_accuracies)) if seed_accuracies else 0.0
        std_acc = float(np.std(seed_accuracies)) if len(seed_accuracies) > 1 else 0.0

        aggregate[strategy_name] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "per_seed_accuracy": {s: a for s, a in zip(seeds, seed_accuracies)},
            "total_examples": len(all_correct),
            "total_correct": sum(all_correct),
            "mean_tokens": float(np.mean(all_tokens)) if all_tokens else 0.0,
            "token_efficiency": compute_token_efficiency(
                [1.0 if c else 0.0 for c in all_correct], all_tokens,
            ) if all_tokens else {},
        }

    baseline_name = "standard_cot"
    if baseline_name in aggregate:
        baseline_correct = []
        for seed in seeds:
            baseline_correct.extend([
                r.get("correct", False)
                for r in all_results.get(baseline_name, {}).get(seed, [])
            ])

        pairwise = {}
        p_values = []
        for strategy_name in strategy_names:
            if strategy_name == baseline_name:
                continue
            strategy_correct = []
            for seed in seeds:
                strategy_correct.extend([
                    r.get("correct", False)
                    for r in all_results.get(strategy_name, {}).get(seed, [])
                ])

            boot = paired_bootstrap_ci(strategy_correct, baseline_correct)
            mcn = mcnemar_test(strategy_correct, baseline_correct)
            cd = cohens_d(
                [1.0 if c else 0.0 for c in strategy_correct],
                [1.0 if c else 0.0 for c in baseline_correct],
            )

            pairwise[f"{strategy_name}_vs_{baseline_name}"] = {
                "bootstrap": boot,
                "mcnemar": mcn,
                "cohens_d": cd,
            }
            p_values.append(boot["p_value"])

        if p_values:
            corrections = bonferroni_correction(p_values)
            for (key, _), correction in zip(
                [(k, v) for k, v in pairwise.items()],
                corrections,
            ):
                pairwise[key]["bonferroni"] = correction

        aggregate["pairwise_vs_baseline"] = pairwise

    correctness_by_strategy = {}
    for strategy_name in strategy_names:
        all_c = []
        for seed in seeds:
            all_c.extend([
                r.get("correct", False)
                for r in all_results.get(strategy_name, {}).get(seed, [])
            ])
        correctness_by_strategy[strategy_name] = all_c

    aggregate["win_loss_matrix"] = per_example_win_loss_matrix(correctness_by_strategy)

    return aggregate


def main():
    parser = argparse.ArgumentParser(description="CLOX: Inference-Time Strategy Selection Experiments")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmarks", type=str, default="gsm8k")
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="11,23,37,47,59")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="Number of GPUs for data-parallel inference")

    args = parser.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)

    if args.n_gpus > 1:
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if args.n_gpus > available_gpus:
                logger.warning(
                    "Requested %d GPUs but only %d available. Clamping to %d.",
                    args.n_gpus, available_gpus, available_gpus,
                )
                args.n_gpus = available_gpus
        else:
            logger.warning("CUDA not available. Falling back to n_gpus=1.")
            args.n_gpus = 1

    logger.info("CLOX Experiment Configuration:")
    logger.info("  Model: %s", args.model)
    logger.info("  Benchmarks: %s", args.benchmarks)
    logger.info("  Strategies: %s", args.strategies)
    logger.info("  Seeds: %s", args.seeds)
    logger.info("  Max examples: %s", args.max_examples)
    logger.info("  Quantize: %s", args.quantize)
    logger.info("  GPUs: %d", args.n_gpus)

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.strategies == "all":
        strategy_names = list(STRATEGY_REGISTRY.keys())
    else:
        strategy_names = [s.strip() for s in args.strategies.split(",")]

    model, tokenizer = None, None
    if args.n_gpus <= 1:
        logger.info("Loading model: %s", args.model)
        model, tokenizer = load_model(args.model, quantize=args.quantize)
        logger.info("Model loaded successfully")

    all_experiment_records = {}
    for benchmark_name in benchmarks:
        logger.info("Loading benchmark: %s", benchmark_name)
        examples = load_benchmark(benchmark_name, max_examples=args.max_examples)
        logger.info("Loaded %d examples from %s", len(examples), benchmark_name)

        record = run_experiment(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            benchmark_name=benchmark_name,
            examples=examples,
            strategy_names=strategy_names,
            seeds=seeds,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
            n_gpus=args.n_gpus,
            quantize=args.quantize,
        )
        all_experiment_records[benchmark_name] = record

    summary_path = os.path.join(args.output_dir, "experiment_summary.json")
    summary = {
        "model": args.model,
        "benchmarks": benchmarks,
        "strategies": strategy_names,
        "seeds": seeds,
        "results": {},
    }
    for bname, record in all_experiment_records.items():
        summary["results"][bname] = {
            sname: record["aggregate"].get(sname, {}).get("mean_accuracy", 0.0)
            for sname in strategy_names
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== FINAL SUMMARY ===")
    for bname in benchmarks:
        logger.info("\n%s:", bname)
        for sname in strategy_names:
            acc = summary["results"].get(bname, {}).get(sname, 0.0)
            logger.info("  %s: %.4f", sname, acc)

    logger.info("\nAll results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
