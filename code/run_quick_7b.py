#!/usr/bin/env python3
"""Quick experiment with 7B model on GSM8K subset.
Validates the full pipeline while waiting for 32B download.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from engine import VLLMEngine
from benchmarks import load_gsm8k, FEW_SHOT_PROMPTS
from evaluation import check_answer
from strategies_v2 import build_strategy
from topology_v2 import batch_estimate_topology

def main():
    OUTPUT = "/home/claude/nips-clox/results/v2_7b"
    os.makedirs(OUTPUT, exist_ok=True)

    print("Loading Qwen2.5-7B-Instruct...")
    engine = VLLMEngine(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=4096,
    )

    examples = load_gsm8k(max_examples=200)
    few_shot = FEW_SHOT_PROMPTS["gsm8k"]
    print(f"Loaded {len(examples)} GSM8K examples")

    # Phase 1: Topology characterization
    print("\n=== Phase 1: Topology (first 100 examples) ===")
    topo_examples = examples[:100]
    questions = [ex.question for ex in topo_examples]
    profiles = batch_estimate_topology(engine, questions, few_shot=few_shot, n_pilot=5, max_tokens=512)

    topo_results = []
    for ex, prof in zip(topo_examples, profiles):
        topo_results.append({
            "example_id": ex.example_id,
            "r_bar": prof.r_bar,
            "epl": prof.epl,
            "n_steps": prof.n_steps,
            "strategy": prof.strategy,
        })
    import numpy as np
    r_bars = [t["r_bar"] for t in topo_results]
    epls = [t["epl"] for t in topo_results]
    print(f"  r̄ = {np.mean(r_bars):.3f} ± {np.std(r_bars):.3f}")
    print(f"  ℓ = {np.mean(epls):.2f} ± {np.std(epls):.2f}")
    strat_dist = {}
    for t in topo_results:
        strat_dist[t["strategy"]] = strat_dist.get(t["strategy"], 0) + 1
    print(f"  Strategy distribution: {strat_dist}")

    with open(os.path.join(OUTPUT, "gsm8k_topology.json"), "w") as f:
        json.dump({"per_example": topo_results, "summary": {
            "r_bar_mean": float(np.mean(r_bars)), "r_bar_std": float(np.std(r_bars)),
            "epl_mean": float(np.mean(epls)), "epl_std": float(np.std(epls)),
            "strategy_distribution": strat_dist,
        }}, f, indent=2)

    # Phase 2: Strategy comparison (200 examples, seed 42)
    print("\n=== Phase 2: Strategy Comparison ===")
    strategy_names = [
        "standard_cot", "self_consistency", "compute_matched_sc",
        "targeted_repair", "random_repair", "backward_cloze",
        "full_regeneration",
    ]

    all_results = {}
    for sname in strategy_names:
        print(f"\n--- {sname} ---")
        t0 = time.time()
        kwargs = {}
        if sname == "self_consistency":
            kwargs = {"k": 5}
        elif sname == "compute_matched_sc":
            kwargs = {}
        strategy = build_strategy(sname, **kwargs)

        results = []
        for i, ex in enumerate(examples):
            try:
                result = strategy.run(engine, ex.question, max_tokens=512, few_shot=few_shot)
                correct = check_answer(result.prediction, ex.answer, ex.answer_type)
                results.append({
                    "example_id": ex.example_id,
                    "prediction": result.prediction,
                    "ground_truth": ex.answer,
                    "correct": correct,
                    "total_tokens": result.total_tokens,
                })
            except Exception as e:
                results.append({"example_id": ex.example_id, "correct": False, "error": str(e)})

            if (i + 1) % 50 == 0:
                acc = sum(r["correct"] for r in results) / len(results)
                print(f"  {i+1}/{len(examples)}: acc={acc:.4f}")

        elapsed = time.time() - t0
        acc = sum(r["correct"] for r in results) / max(len(results), 1)
        tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens", 0) > 0]
        print(f"  Final: acc={acc:.4f}, mean_tokens={np.mean(tokens):.0f}, time={elapsed:.0f}s")
        all_results[sname] = results

    # Save results
    summary = {}
    for sname, results in all_results.items():
        acc = sum(r["correct"] for r in results) / max(len(results), 1)
        tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens", 0) > 0]
        summary[sname] = {
            "accuracy": acc,
            "mean_tokens": float(np.mean(tokens)) if tokens else 0,
            "n_examples": len(results),
        }

    with open(os.path.join(OUTPUT, "gsm8k_strategies.json"), "w") as f:
        json.dump({"summary": summary, "per_strategy": all_results}, f, indent=2, default=str)

    print("\n=== FINAL RESULTS ===")
    print(f"{'Strategy':<25s} {'Acc':>8s} {'Tokens':>8s}")
    print("-" * 45)
    for sname in strategy_names:
        s = summary[sname]
        print(f"{sname:<25s} {s['accuracy']:>8.4f} {s['mean_tokens']:>8.0f}")

    print(f"\nResults saved to {OUTPUT}/")

if __name__ == "__main__":
    main()
