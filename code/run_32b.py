#!/usr/bin/env python3
"""Full experiment with Qwen3-32B on all benchmarks.
Designed for 2×H100 with tensor parallel.
"""
import sys, os, json, time, logging
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engine import VLLMEngine
from benchmarks import load_benchmark, FEW_SHOT_PROMPTS
from evaluation import check_answer, paired_bootstrap_ci, mcnemar_test, cohens_d, compute_token_efficiency
from strategies_v2 import build_strategy, STRATEGY_REGISTRY
from topology_v2 import batch_estimate_topology

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clox32b")

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "v2_32b")

def run_topology(engine, benchmark_name, examples, few_shot, n_pilot=5):
    """Phase 1: Topology characterization."""
    log.info(f"Phase 1: Topology for {benchmark_name} ({len(examples)} examples)")
    questions = [ex.question for ex in examples]

    # Small batches to avoid OOM (batch × n_pilot sequences)
    batch_size = 10
    all_profiles = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        log.info(f"  Batch {i//batch_size + 1}/{(len(questions) + batch_size - 1) // batch_size}")
        profiles = batch_estimate_topology(engine, batch, few_shot=few_shot, n_pilot=n_pilot, max_tokens=512)
        all_profiles.extend(profiles)

    results = []
    for ex, prof in zip(examples, all_profiles):
        results.append({
            "example_id": ex.example_id, "difficulty": ex.difficulty,
            "r_bar": prof.r_bar, "epl": prof.epl,
            "n_steps": prof.n_steps, "strategy": prof.strategy,
        })

    r_bars = [r["r_bar"] for r in results]
    epls = [r["epl"] for r in results]
    strat_dist = {}
    for r in results:
        strat_dist[r["strategy"]] = strat_dist.get(r["strategy"], 0) + 1

    summary = {
        "benchmark": benchmark_name, "n_examples": len(results),
        "r_bar_mean": float(np.mean(r_bars)), "r_bar_std": float(np.std(r_bars)),
        "epl_mean": float(np.mean(epls)), "epl_std": float(np.std(epls)),
        "strategy_distribution": strat_dist,
    }
    log.info(f"  r̄={summary['r_bar_mean']:.3f}±{summary['r_bar_std']:.3f}, ℓ={summary['epl_mean']:.2f}±{summary['epl_std']:.2f}")
    log.info(f"  Strategies: {strat_dist}")

    bm_dir = os.path.join(OUTPUT, benchmark_name)
    os.makedirs(bm_dir, exist_ok=True)
    with open(os.path.join(bm_dir, "topology.json"), "w") as f:
        json.dump({"summary": summary, "per_example": results}, f, indent=2)
    return summary


def run_strategies(engine, benchmark_name, examples, few_shot, strategy_names, seeds):
    """Phase 2: Strategy comparison."""
    log.info(f"Phase 2: Strategies for {benchmark_name} ({len(examples)} ex × {len(strategy_names)} strats × {len(seeds)} seeds)")
    bm_dir = os.path.join(OUTPUT, benchmark_name)
    os.makedirs(bm_dir, exist_ok=True)

    all_results = {}
    for sname in strategy_names:
        all_results[sname] = {}
        kwargs = {}
        if sname == "self_consistency":
            kwargs = {"k": 8}

        for seed in seeds:
            ckpt = os.path.join(bm_dir, f".ckpt_{sname}_s{seed}.json")
            cached = []
            if os.path.exists(ckpt):
                with open(ckpt) as f:
                    cached = json.load(f)
            done_ids = {r["example_id"] for r in cached}
            remaining = [ex for ex in examples if ex.example_id not in done_ids]

            if cached and remaining:
                log.info(f"  Resuming {sname}/s{seed}: {len(cached)} done, {len(remaining)} left")

            strategy = build_strategy(sname, **kwargs)
            results = list(cached)

            for i, ex in enumerate(remaining):
                try:
                    result = strategy.run(engine, ex.question, max_tokens=512, few_shot=few_shot)
                    correct = check_answer(result.prediction, ex.answer, ex.answer_type)
                    results.append({
                        "example_id": ex.example_id,
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
                    results.append({"example_id": ex.example_id, "correct": False, "error": str(e), "seed": seed})

                if (i + 1) % 100 == 0:
                    acc = sum(r.get("correct", False) for r in results) / len(results)
                    log.info(f"  {sname} s{seed}: {len(results)}/{len(examples)} acc={acc:.4f}")
                    with open(ckpt, "w") as f:
                        json.dump(results, f, default=str)

            with open(ckpt, "w") as f:
                json.dump(results, f, default=str)
            all_results[sname][seed] = results

            acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
            tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens", 0) > 0]
            log.info(f"  {sname} s{seed}: acc={acc:.4f} tokens={np.mean(tokens):.0f}")

    # Aggregate
    aggregate = {}
    for sname in strategy_names:
        seed_accs = []
        all_c, all_t = [], []
        for seed in seeds:
            res = all_results[sname].get(seed, [])
            c = [r.get("correct", False) for r in res]
            seed_accs.append(sum(c) / max(len(c), 1))
            all_c.extend(c)
            all_t.extend(r.get("total_tokens", 0) for r in res)
        aggregate[sname] = {
            "mean_accuracy": float(np.mean(seed_accs)),
            "std_accuracy": float(np.std(seed_accs)) if len(seed_accs) > 1 else 0,
            "per_seed": {s: a for s, a in zip(seeds, seed_accs)},
            "mean_tokens": float(np.mean(all_t)) if all_t else 0,
        }

    # Pairwise stats
    baseline = "standard_cot"
    if baseline in aggregate:
        pairwise = {}
        for sname in strategy_names:
            if sname == baseline: continue
            s_c = [r.get("correct", False) for seed in seeds for r in all_results.get(sname, {}).get(seed, [])]
            b_c = [r.get("correct", False) for seed in seeds for r in all_results.get(baseline, {}).get(seed, [])]
            boot = paired_bootstrap_ci(s_c, b_c)
            pairwise[f"{sname}_vs_{baseline}"] = {"bootstrap": boot}
        aggregate["pairwise"] = pairwise

    with open(os.path.join(bm_dir, "strategies.json"), "w") as f:
        json.dump({"aggregate": aggregate, "strategies": strategy_names, "seeds": seeds}, f, indent=2, default=str)

    # Clean checkpoints
    for p in os.listdir(bm_dir):
        if p.startswith(".ckpt_"):
            os.unlink(os.path.join(bm_dir, p))

    return aggregate


def run_adaptive(engine, benchmark_name, examples, few_shot, seeds):
    """Phase 3: CLOX-Adaptive evaluation."""
    log.info(f"Phase 3: Adaptive for {benchmark_name}")
    bm_dir = os.path.join(OUTPUT, benchmark_name)
    os.makedirs(bm_dir, exist_ok=True)

    adaptive = build_strategy("clox_adaptive")
    all_results = {}

    for seed in seeds:
        results = []
        for i, ex in enumerate(examples):
            try:
                result = adaptive.run(engine, ex.question, max_tokens=512, few_shot=few_shot)
                correct = check_answer(result.prediction, ex.answer, ex.answer_type)
                results.append({
                    "example_id": ex.example_id, "correct": correct,
                    "total_tokens": result.total_tokens,
                    "selected": result.strategy_name,
                    "topology": {"r_bar": result.topology.r_bar if result.topology else None,
                                 "epl": result.topology.epl if result.topology else None},
                    "seed": seed,
                })
            except Exception as e:
                results.append({"example_id": ex.example_id, "correct": False, "error": str(e), "seed": seed})

            if (i + 1) % 100 == 0:
                acc = sum(r.get("correct", False) for r in results) / len(results)
                log.info(f"  adaptive s{seed}: {i+1}/{len(examples)} acc={acc:.4f}")

        all_results[seed] = results
        acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
        log.info(f"  adaptive s{seed}: acc={acc:.4f}")

    with open(os.path.join(bm_dir, "adaptive.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results


def main():
    log.info("=== CLOX v2: Full Experiment with Qwen3-32B ===")
    log.info(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    engine = VLLMEngine(
        model_name="Qwen/Qwen3-32B-AWQ",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        quantization="awq",
    )

    benchmarks = ["gsm8k", "math", "strategyqa", "arc_challenge"]
    strategy_names = [
        "standard_cot", "self_consistency", "compute_matched_sc",
        "targeted_repair", "random_repair", "backward_cloze",
        "full_regeneration", "hierarchical_repair",
    ]
    seeds = [11, 23, 37, 47, 59]

    for bname in benchmarks:
        log.info(f"\n{'='*60}")
        log.info(f"Benchmark: {bname}")
        log.info(f"{'='*60}")

        max_ex = None  # Full benchmark
        if bname == "math":
            max_ex = 500  # Subset for speed
        examples = load_benchmark(bname, max_examples=max_ex)
        few_shot = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")
        log.info(f"Loaded {len(examples)} examples")

        # Phase 1
        run_topology(engine, bname, examples[:200], few_shot, n_pilot=8)

        # Phase 2: Single seed first for speed, then remaining seeds
        run_strategies(engine, bname, examples, few_shot, strategy_names, seeds[:1])

        # Phase 3
        run_adaptive(engine, bname, examples[:200], few_shot, seeds[:1])

    # Phase 2 remaining seeds (most expensive - do last)
    log.info("\n=== Remaining seeds (Phase 2) ===")
    for bname in benchmarks:
        max_ex = 500 if bname == "math" else None
        examples = load_benchmark(bname, max_examples=max_ex)
        few_shot = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")
        run_strategies(engine, bname, examples, few_shot, strategy_names, seeds[1:])

    log.info(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Results in: {OUTPUT}")

    # Print summary
    log.info("\n=== FINAL SUMMARY ===")
    for bname in benchmarks:
        bm_dir = os.path.join(OUTPUT, bname)
        strat_file = os.path.join(bm_dir, "strategies.json")
        if os.path.exists(strat_file):
            with open(strat_file) as f:
                d = json.load(f)
            log.info(f"\n{bname}:")
            for sname in d.get("strategies", []):
                acc = d.get("aggregate", {}).get(sname, {}).get("mean_accuracy", 0)
                log.info(f"  {sname}: {acc:.4f}")


if __name__ == "__main__":
    main()
