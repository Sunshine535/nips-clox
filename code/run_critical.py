#!/usr/bin/env python3
"""Critical remaining experiments for NeurIPS submission.

Phase A: GSM8K CLOX-Adaptive (200 ex × 3 seeds)
Phase B: StrategyQA full comparison (200 ex × 6 strategies × 3 seeds)
Phase C: Proxy validation on GSM8K (50 ex)
Phase D: Compute paired bootstrap + McNemar on all GSM8K results
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from benchmarks import load_benchmark, FEW_SHOT_PROMPTS
from engine import VLLMEngine
from evaluation import check_answer, paired_bootstrap_ci, mcnemar_test, cohens_d
from strategies_v2 import build_strategy
from topology_v2 import batch_estimate_topology

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clox_critical")

OUTPUT = "/home/claude/nips-clox/results/v3/Qwen2.5-32B-Instruct-AWQ"
SEEDS = [11, 23, 37]


def _ckpt(d, tag):
    return os.path.join(d, f".ckpt_{tag}.json")

def _load(p):
    if os.path.isfile(p):
        with open(p) as f: return json.load(f)
    return []

def _save(p, d):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p + ".tmp", "w") as f: json.dump(d, f, default=str)
    os.replace(p + ".tmp", p)


def run_strategy_on_benchmark(engine, bname, strategy_names, seeds, max_ex, output_dir):
    """Run strategies on a benchmark with checkpointing."""
    examples = load_benchmark(bname, max_examples=max_ex)
    few_shot = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")
    log.info(f"  {bname}: {len(examples)} examples, strategies={strategy_names}")

    os.makedirs(output_dir, exist_ok=True)

    for sname in strategy_names:
        kwargs = {"k": 5} if sname == "self_consistency" else {}
        strategy = build_strategy(sname, **kwargs)

        for seed in seeds:
            ckpt = _ckpt(output_dir, f"{sname}_s{seed}")
            cached = _load(ckpt)
            done_ids = {r["example_id"] for r in cached}
            remaining = [ex for ex in examples if ex.example_id not in done_ids]

            if not remaining and cached:
                acc = sum(r.get("correct", False) for r in cached) / max(len(cached), 1)
                log.info(f"    {sname}/s{seed}: done ({acc:.4f})")
                continue

            if cached:
                log.info(f"    Resuming {sname}/s{seed}: {len(cached)} done, {len(remaining)} left")

            results = list(cached)
            for i, ex in enumerate(remaining):
                try:
                    result = strategy.run(engine, ex.question, max_tokens=512, few_shot=few_shot)
                    correct = check_answer(result.prediction, ex.answer, ex.answer_type)
                    results.append({
                        "example_id": ex.example_id,
                        "benchmark": bname,
                        "difficulty": ex.difficulty,
                        "prediction": result.prediction,
                        "ground_truth": ex.answer,
                        "correct": correct,
                        "total_tokens": result.total_tokens,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "strategy": result.strategy_name,
                        "seed": seed,
                    })
                except Exception as e:
                    log.warning(f"    Error {sname}/{ex.example_id}: {e}")
                    results.append({"example_id": ex.example_id, "correct": False,
                                    "total_tokens": 0, "error": str(e), "seed": seed})

                if (i + 1) % 50 == 0:
                    acc = sum(r.get("correct", False) for r in results) / len(results)
                    log.info(f"      {sname}/s{seed}: {len(results)}/{len(examples)} acc={acc:.4f}")
                    _save(ckpt, results)

            _save(ckpt, results)
            acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
            tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens", 0) > 0]
            log.info(f"    {sname}/s{seed}: acc={acc:.4f} tokens={np.mean(tokens):.0f}")


def run_proxy_validation(engine, bname, output_dir, n_gt=50):
    """Proxy validation: compare M=3,5,8 pilot estimates against M=30 ground truth."""
    examples = load_benchmark(bname, max_examples=n_gt)
    few_shot = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")
    questions = [ex.question for ex in examples]
    log.info(f"  Proxy validation on {bname}: {len(examples)} examples")

    from scipy.stats import spearmanr

    # Ground truth: 30 pilot traces
    log.info("    Computing ground truth (M=30)...")
    gt = batch_estimate_topology(engine, questions, few_shot=few_shot, n_pilot=30, max_tokens=512)
    gt_r = [p.r_bar for p in gt]
    gt_l = [p.epl for p in gt]

    validation = {"gt": {"r_bar_mean": float(np.mean(gt_r)), "r_bar_std": float(np.std(gt_r)),
                         "epl_mean": float(np.mean(gt_l)), "epl_std": float(np.std(gt_l))}}

    for m in [3, 5, 8]:
        log.info(f"    Computing pilot (M={m})...")
        pilot = batch_estimate_topology(engine, questions, few_shot=few_shot, n_pilot=m, max_tokens=512)
        p_r = [p.r_bar for p in pilot]
        p_l = [p.epl for p in pilot]

        rho_r, pval_r = spearmanr(gt_r, p_r)
        rho_l, pval_l = spearmanr(gt_l, p_l)

        # Strategy agreement
        gt_strats = [p.strategy for p in gt]
        p_strats = [p.strategy for p in pilot]
        agree = sum(1 for a, b in zip(gt_strats, p_strats) if a == b) / len(gt_strats)

        validation[f"M={m}"] = {
            "r_bar_mean": float(np.mean(p_r)), "r_bar_std": float(np.std(p_r)),
            "epl_mean": float(np.mean(p_l)), "epl_std": float(np.std(p_l)),
            "rho_r": float(rho_r), "rho_r_p": float(pval_r),
            "rho_l": float(rho_l), "rho_l_p": float(pval_l),
            "r_bar_mae": float(np.mean(np.abs(np.array(gt_r) - np.array(p_r)))),
            "epl_mae": float(np.mean(np.abs(np.array(gt_l) - np.array(p_l)))),
            "strategy_agreement": float(agree),
        }
        log.info(f"      M={m}: ρ(r̄)={rho_r:.3f} ρ(ℓ)={rho_l:.3f} strat_agree={agree:.2f}")

    with open(os.path.join(output_dir, "proxy_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)
    return validation


def compute_statistics(output_dir, bname, strategies, seeds):
    """Compute paired bootstrap CI and McNemar for all pairs vs baseline."""
    log.info(f"  Computing statistics for {bname}...")

    all_results = {}
    for sname in strategies:
        all_results[sname] = {}
        for seed in seeds:
            ckpt = _ckpt(os.path.join(output_dir, bname), f"{sname}_s{seed}")
            all_results[sname][seed] = _load(ckpt)

    # Aggregate
    aggregate = {}
    for sname in strategies:
        seed_accs = []
        all_correct = []
        all_tokens = []
        for seed in seeds:
            res = all_results[sname].get(seed, [])
            correct = [r.get("correct", False) for r in res]
            seed_accs.append(sum(correct) / max(len(correct), 1))
            all_correct.extend(correct)
            all_tokens.extend(r.get("total_tokens", 0) for r in res if r.get("total_tokens", 0) > 0)

        aggregate[sname] = {
            "mean_accuracy": float(np.mean(seed_accs)) if seed_accs else 0,
            "std_accuracy": float(np.std(seed_accs)) if len(seed_accs) > 1 else 0,
            "per_seed": {s: a for s, a in zip(seeds, seed_accs)},
            "mean_tokens": float(np.mean(all_tokens)) if all_tokens else 0,
            "n_total": len(all_correct),
        }

    # Pairwise stats
    baseline = "standard_cot"
    pairwise = {}
    for sname in strategies:
        if sname == baseline:
            continue
        s_c = [r.get("correct", False) for seed in seeds for r in all_results.get(sname, {}).get(seed, [])]
        b_c = [r.get("correct", False) for seed in seeds for r in all_results.get(baseline, {}).get(seed, [])]
        if s_c and b_c and len(s_c) == len(b_c):
            boot = paired_bootstrap_ci(s_c, b_c)
            mcn = mcnemar_test(s_c, b_c)
            cd = cohens_d([1.0 if c else 0.0 for c in s_c], [1.0 if c else 0.0 for c in b_c])
            pairwise[f"{sname}_vs_{baseline}"] = {"bootstrap": boot, "mcnemar": mcn, "cohens_d": cd}

    # Also: targeted vs random, targeted vs full_regen, targeted vs SC
    key_pairs = [
        ("targeted_repair", "random_repair"),
        ("targeted_repair", "full_regeneration"),
        ("targeted_repair", "self_consistency"),
        ("self_consistency", "compute_matched_sc"),
    ]
    for a, b in key_pairs:
        if a in all_results and b in all_results:
            a_c = [r.get("correct", False) for seed in seeds for r in all_results.get(a, {}).get(seed, [])]
            b_c = [r.get("correct", False) for seed in seeds for r in all_results.get(b, {}).get(seed, [])]
            if a_c and b_c and len(a_c) == len(b_c):
                boot = paired_bootstrap_ci(a_c, b_c)
                mcn = mcnemar_test(a_c, b_c)
                cd = cohens_d([1.0 if c else 0.0 for c in a_c], [1.0 if c else 0.0 for c in b_c])
                pairwise[f"{a}_vs_{b}"] = {"bootstrap": boot, "mcnemar": mcn, "cohens_d": cd}

    aggregate["pairwise"] = pairwise

    with open(os.path.join(output_dir, bname, "statistics.json"), "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    # Print summary
    log.info(f"\n  === {bname} Statistics ===")
    for sname in strategies:
        a = aggregate[sname]
        log.info(f"    {sname}: {a['mean_accuracy']:.4f} ± {a['std_accuracy']:.4f} ({a['mean_tokens']:.0f} tok)")
    log.info(f"\n  Pairwise tests:")
    for pair, stats in pairwise.items():
        boot = stats.get("bootstrap", {})
        mcn = stats.get("mcnemar", {})
        cd = stats.get("cohens_d", 0)
        ci = boot.get("ci_95", [0, 0])
        p = mcn.get("p_value", 1.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log.info(f"    {pair}: Δ={boot.get('mean_diff', 0):.4f} CI95=[{ci[0]:.4f},{ci[1]:.4f}] p={p:.4f} d={cd:.3f} {sig}")

    return aggregate


def main():
    log.info("=== CLOX Critical Experiments ===")

    engine = VLLMEngine(
        model_name="Qwen/Qwen2.5-32B-Instruct-AWQ",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        quantization="awq",
    )
    log.info("Model loaded!")

    # Phase A: GSM8K CLOX-Adaptive (skip hierarchical)
    log.info("\n=== Phase A: GSM8K CLOX-Adaptive ===")
    run_strategy_on_benchmark(
        engine, "gsm8k",
        strategy_names=["clox_adaptive"],
        seeds=SEEDS, max_ex=200,
        output_dir=os.path.join(OUTPUT, "gsm8k"),
    )

    # Phase B: StrategyQA full comparison
    log.info("\n=== Phase B: StrategyQA Full ===")
    sq_strategies = ["standard_cot", "self_consistency", "targeted_repair",
                     "random_repair", "full_regeneration", "clox_adaptive"]
    run_strategy_on_benchmark(
        engine, "strategyqa",
        strategy_names=sq_strategies,
        seeds=SEEDS, max_ex=200,
        output_dir=os.path.join(OUTPUT, "strategyqa"),
    )

    # Phase C: Proxy validation
    log.info("\n=== Phase C: Proxy Validation ===")
    run_proxy_validation(engine, "gsm8k", OUTPUT, n_gt=50)

    # Phase D: Statistics (no GPU needed but runs after above)
    log.info("\n=== Phase D: Statistics ===")
    gsm8k_strats = ["standard_cot", "self_consistency", "compute_matched_sc",
                     "targeted_repair", "random_repair", "backward_cloze",
                     "full_regeneration", "clox_adaptive"]
    compute_statistics(OUTPUT, "gsm8k", gsm8k_strats, SEEDS)
    compute_statistics(OUTPUT, "strategyqa", sq_strategies, SEEDS)

    log.info("\n=== ALL CRITICAL EXPERIMENTS COMPLETE ===")


if __name__ == "__main__":
    main()
