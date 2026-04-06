#!/usr/bin/env python3
"""Quick smoke test: verify vLLM + strategies work on a few examples."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from engine import VLLMEngine, extract_answer
from strategies_v2 import build_strategy, STRATEGY_REGISTRY
from benchmarks import load_gsm8k

def main():
    print("Loading model...")
    engine = VLLMEngine(
        model_name="Qwen/Qwen3-32B",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )
    print("Model loaded!")

    # Test with 5 GSM8K examples
    examples = load_gsm8k(max_examples=5)
    print(f"\nLoaded {len(examples)} GSM8K examples")

    few_shot = (
        "Q: There are 15 trees in the grove. Grove workers will plant trees today. "
        "After they are done, there will be 21 trees. How many trees did the workers plant today?\n"
        "A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after planting. "
        "So 21 - 15 = 6 trees were planted. The answer is 6.\n"
    )

    # Test each strategy on first example
    for name in ["standard_cot", "self_consistency", "targeted_repair", "random_repair", "backward_cloze"]:
        print(f"\n--- {name} ---")
        strategy = build_strategy(name, **({"k": 3} if name == "self_consistency" else {}))
        result = strategy.run(engine, examples[0].question, max_tokens=512, few_shot=few_shot)
        print(f"  Prediction: {result.prediction}")
        print(f"  Ground truth: {examples[0].answer}")
        from evaluation import check_answer
        correct = check_answer(result.prediction, examples[0].answer, examples[0].answer_type)
        print(f"  Correct: {correct}")
        print(f"  Tokens: {result.total_tokens} (prompt={result.prompt_tokens}, comp={result.completion_tokens})")

    # Test topology estimation
    from topology_v2 import estimate_topology
    print("\n--- Topology Estimation ---")
    topo = estimate_topology(engine, examples[0].question, few_shot=few_shot, n_pilot=5, do_regeneration_test=False)
    print(f"  r̄ = {topo.r_bar:.3f}")
    print(f"  ℓ = {topo.epl:.2f}")
    print(f"  n_steps = {topo.n_steps:.1f}")
    print(f"  Recommended: {topo.strategy}")

    print("\n=== SMOKE TEST PASSED ===")

if __name__ == "__main__":
    main()
