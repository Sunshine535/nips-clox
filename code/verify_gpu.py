#!/usr/bin/env python3
"""Quick verification: auto-detect GPUs, load model, run one inference."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from engine import detect_gpu_count, auto_tp, VLLMEngine


def main():
    n_gpus = detect_gpu_count()
    model = os.environ.get("MODEL", "Qwen/Qwen3.5-27B")
    tp = auto_tp(model, n_gpus)
    print(f"[verify] GPUs: {n_gpus}, model: {model}, TP: {tp}")

    t0 = time.time()
    engine = VLLMEngine(model, tensor_parallel_size=tp)
    print(f"[verify] Model loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    out = engine.generate_single("What is 2+3? Answer with just the number.", max_tokens=32, logprobs=5)
    print(f"[verify] Inference: {out.text!r} ({time.time()-t1:.2f}s)")
    print(f"[verify] Tokens: prompt={out.prompt_tokens}, completion={out.completion_tokens}")

    # Batch test
    t2 = time.time()
    batch_out = engine.generate_batch(
        ["What is 7*8?", "What is the capital of France?", "2+2=?"],
        max_tokens=32, logprobs=5,
    )
    print(f"[verify] Batch(3): {[o.text[:30] for o in batch_out]} ({time.time()-t2:.2f}s)")
    print(f"[verify] SUCCESS — TP={tp} on {n_gpus} GPUs")


if __name__ == "__main__":
    main()
