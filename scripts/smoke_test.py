#!/usr/bin/env python3
"""Quick smoke test: 20 GSM8K examples, all strategies, 1 seed, verify differentiation."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import torch
from benchmarks import load_gsm8k, FEW_SHOT_PROMPTS
from strategies import STRATEGY_REGISTRY, build_strategy
from evaluation import check_answer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
N_EXAMPLES = 20
SEED = 42
MAX_NEW_TOKENS = 512
DEVICE = "cuda:0"


def main():
    print(f"=== CLOX Smoke Test ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Examples: {N_EXAMPLES}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"GPU memory used: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    print()

    # Load benchmark
    print("Loading GSM8K...")
    examples = load_gsm8k(max_examples=N_EXAMPLES)
    few_shot = FEW_SHOT_PROMPTS.get("gsm8k", "")
    print(f"Loaded {len(examples)} examples")
    print()

    # Test each strategy
    strategies_to_test = [
        "standard_cot",
        "self_consistency",
        "backward_cloze",
        "uncertainty_masked_repair",
        "random_masked_repair",
        "full_regeneration",
        "hierarchical_repair",
        "clox_adaptive",
    ]

    results = {}
    for strategy_name in strategies_to_test:
        print(f"\n--- Testing: {strategy_name} ---")
        torch.manual_seed(SEED)
        strategy = build_strategy(strategy_name)

        correct = 0
        total_tokens = 0
        predictions = []
        t_start = time.time()

        for i, ex in enumerate(examples):
            try:
                result = strategy.run(
                    model=model,
                    tokenizer=tokenizer,
                    question=ex.question,
                    max_new_tokens=MAX_NEW_TOKENS,
                    few_shot_prompt=few_shot,
                )
                is_correct = check_answer(result.prediction, ex.answer, ex.answer_type)
                correct += int(is_correct)
                total_tokens += result.total_tokens
                predictions.append(result.prediction)

                if (i + 1) % 5 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{i+1}/{N_EXAMPLES}] acc={correct/(i+1):.2%} tokens/ex={total_tokens/(i+1):.0f} elapsed={elapsed:.1f}s")
            except Exception as e:
                print(f"  ERROR on example {i}: {e}")
                predictions.append("ERROR")

        elapsed = time.time() - t_start
        acc = correct / N_EXAMPLES
        results[strategy_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": N_EXAMPLES,
            "mean_tokens": total_tokens / max(N_EXAMPLES, 1),
            "elapsed_s": elapsed,
            "predictions": predictions,
        }
        print(f"  RESULT: {strategy_name} = {acc:.2%} ({correct}/{N_EXAMPLES}), "
              f"mean_tokens={total_tokens/N_EXAMPLES:.0f}, time={elapsed:.1f}s")

    # Summary
    print("\n\n=== SMOKE TEST SUMMARY ===")
    print(f"{'Strategy':<35} {'Accuracy':>10} {'Tokens/Ex':>10} {'Time':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<35} {r['accuracy']:>9.2%} {r['mean_tokens']:>10.0f} {r['elapsed_s']:>7.1f}s")

    # Check differentiation: are predictions different across strategies?
    print("\n=== PREDICTION DIFFERENTIATION ===")
    for i in range(min(3, N_EXAMPLES)):
        print(f"\nExample {i}: ground_truth={examples[i].answer}")
        for name in strategies_to_test:
            pred = results[name]["predictions"][i]
            correct_mark = "✓" if check_answer(pred, examples[i].answer, examples[i].answer_type) else "✗"
            print(f"  {name:<35} → {pred:<20} {correct_mark}")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "smoke_test_h100")
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "smoke_results.json")
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != "predictions"} for k, v in results.items()}
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
