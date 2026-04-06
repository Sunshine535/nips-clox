#!/usr/bin/env python3
"""Quick smoke test with cached 7B model."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from engine import VLLMEngine, extract_answer
from strategies_v2 import build_strategy
from benchmarks import load_gsm8k
from evaluation import check_answer
from topology_v2 import estimate_topology

def main():
    print("Loading Qwen2.5-7B-Instruct...")
    engine = VLLMEngine(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=4096,
    )
    print("Model loaded!")

    examples = load_gsm8k(max_examples=3)
    print(f"Loaded {len(examples)} examples")

    few_shot = (
        "Q: There are 15 trees in the grove. After planting, there will be 21. "
        "How many trees did the workers plant?\n"
        "A: Let's think step by step. 21 - 15 = 6. The answer is 6.\n"
    )

    for name in ["standard_cot", "targeted_repair", "backward_cloze"]:
        print(f"\n--- {name} ---")
        strategy = build_strategy(name)
        result = strategy.run(engine, examples[0].question, max_tokens=512, few_shot=few_shot)
        correct = check_answer(result.prediction, examples[0].answer, examples[0].answer_type)
        print(f"  Pred: {result.prediction} | GT: {examples[0].answer} | Correct: {correct}")
        print(f"  Tokens: total={result.total_tokens}, prompt={result.prompt_tokens}, comp={result.completion_tokens}")

    print("\n--- Topology ---")
    topo = estimate_topology(engine, examples[0].question, few_shot=few_shot, n_pilot=5, do_regeneration_test=False)
    print(f"  r̄={topo.r_bar:.3f}, ℓ={topo.epl:.2f}, n_steps={topo.n_steps:.1f}, strategy={topo.strategy}")

    print("\n=== SMOKE TEST PASSED ===")

if __name__ == "__main__":
    main()
