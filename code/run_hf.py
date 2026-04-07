#!/usr/bin/env python3
"""Experiment runner using HuggingFace generate (fallback for constrained GPU memory)."""
import sys, os, json, time, logging, math, re
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import load_benchmark, FEW_SHOT_PROMPTS
from evaluation import check_answer, paired_bootstrap_ci, compute_token_efficiency

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clox_hf")

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "v2_32b")


def load_model(model_name):
    log.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    log.info(f"Loaded on {model.device}")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=512, temperature=0.0, do_sample=False, logprobs=True):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": logprobs,
    }
    if do_sample and temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(input_ids, **gen_kwargs)

    gen_ids = outputs.sequences[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    comp_tokens = len(gen_ids)

    lps = []
    top_lps = []
    if logprobs and hasattr(outputs, "scores") and outputs.scores:
        for idx, score in enumerate(outputs.scores):
            if idx >= len(gen_ids):
                break
            log_p = F.log_softmax(score[0].float(), dim=-1)
            token_id = gen_ids[idx]
            lps.append(float(log_p[token_id].cpu()))
            # Top-20 for entropy
            top_vals, top_ids = torch.topk(log_p, 20)
            top_lps.append({int(tid): float(tv) for tid, tv in zip(top_ids.cpu(), top_vals.cpu())})

    return text, lps, top_lps, prompt_len, comp_tokens


def extract_answer(text):
    patterns = [
        r"\\boxed\{([^}]+)\}", r"####\s*(.+)",
        r"(?:final answer|the answer is|answer:)\s*[:\s]*([^\n.;]+)",
        r"(?:Therefore|So|Thus|Hence),?\s*(?:the (?:final )?answer is)?\s*([^\n.;]+)",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            ans = matches[-1].group(1).strip()
            if 0 < len(ans) < 200:
                return ans
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        num = re.search(r"[-−]?\d+(?:\.\d+)?", line)
        if num:
            return num.group()
        if len(line) < 50:
            return line
    return lines[-1].strip() if lines else ""


def split_steps(text):
    parts = re.split(r'\n(?=(?:Step\s+)?\d+[\.\):])', text)
    if len(parts) >= 3:
        return [s.strip() for s in parts if s.strip() and len(s.split()) >= 3]
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in parts if s.strip() and len(s.split()) >= 3]


def step_entropy(lps_list, top_lps_list, steps, n_tokens):
    """Compute per-step entropy from top-K logprobs."""
    if not steps or not top_lps_list:
        return []
    tps = max(n_tokens // len(steps), 1)
    ents = []
    for i in range(len(steps)):
        start = i * tps
        end = min((i + 1) * tps, len(top_lps_list))
        step_ent = []
        for pos in range(start, end):
            if pos < len(top_lps_list):
                lps_dict = top_lps_list[pos]
                probs = [math.exp(lp) for lp in lps_dict.values()]
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]
                    ent = -sum(p * math.log(max(p, 1e-30)) for p in probs if p > 0)
                    step_ent.append(ent)
        ents.append(float(np.mean(step_ent)) if step_ent else 0.0)
    return ents


# ── Strategies ���──────────────────────────────────��──────────────────

def run_standard_cot(model, tokenizer, question, few_shot, max_tokens=512):
    prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step.\n"
    text, lps, top_lps, ptok, ctok = generate(model, tokenizer, prompt, max_tokens)
    return extract_answer(text), ptok + ctok, ptok, ctok, text

def run_self_consistency(model, tokenizer, question, few_shot, k=8, max_tokens=512):
    prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step.\n"
    answers = []
    total_ptok, total_ctok = 0, 0
    for _ in range(k):
        text, lps, top_lps, ptok, ctok = generate(model, tokenizer, prompt, max_tokens, temperature=0.7, do_sample=True)
        answers.append(extract_answer(text))
        total_ptok += ptok
        total_ctok += ctok
    majority = Counter(answers).most_common(1)[0][0]
    return majority, total_ptok + total_ctok, total_ptok, total_ctok, f"SC answers: {answers}"

def run_compute_matched_sc(model, tokenizer, question, few_shot, max_tokens=512):
    return run_self_consistency(model, tokenizer, question, few_shot, k=2, max_tokens=max_tokens)

def run_targeted_repair(model, tokenizer, question, few_shot, max_tokens=512, mask_frac=0.4):
    prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step.\n"
    text, lps, top_lps, ptok, ctok = generate(model, tokenizer, prompt, max_tokens)
    steps = split_steps(text)
    if len(steps) < 2:
        return extract_answer(text), ptok + ctok, ptok, ctok, text

    ents = step_entropy(lps, top_lps, steps, len(lps))
    n_mask = max(1, int(len(steps) * mask_frac))
    ranked = sorted(range(len(ents)), key=lambda i: ents[i], reverse=True)
    masked = sorted(ranked[:n_mask])

    visible = []
    for i, s in enumerate(steps):
        if i in masked:
            visible.append(f"Step {i+1}: [HIGH UNCERTAINTY - NEEDS CORRECTION]")
        else:
            visible.append(f"Step {i+1}: {s}")

    repair_prompt = (
        f"Question: {question}\n\nA previous solution had uncertain steps:\n\n"
        + "\n".join(visible)
        + "\n\nRewrite the solution correcting uncertain steps. End with: The answer is <answer>.\n"
    )
    r_text, r_lps, r_top_lps, r_ptok, r_ctok = generate(model, tokenizer, repair_prompt, max_tokens, temperature=0.3, do_sample=True)
    answer = extract_answer(r_text) or extract_answer(text)
    return answer, ptok + ctok + r_ptok + r_ctok, ptok + r_ptok, ctok + r_ctok, f"FIRST:\n{text}\nREPAIR:\n{r_text}"

def run_random_repair(model, tokenizer, question, few_shot, max_tokens=512, mask_frac=0.4):
    prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step.\n"
    text, lps, top_lps, ptok, ctok = generate(model, tokenizer, prompt, max_tokens)
    steps = split_steps(text)
    if len(steps) < 2:
        return extract_answer(text), ptok + ctok, ptok, ctok, text

    rng = np.random.default_rng(hash(question) % (2**32) + 12345)
    n_mask = max(1, int(len(steps) * mask_frac))
    masked = sorted(rng.choice(len(steps), size=min(n_mask, len(steps)), replace=False).tolist())

    visible = []
    for i, s in enumerate(steps):
        if i in masked:
            visible.append(f"Step {i+1}: [MASKED - fill in]")
        else:
            visible.append(f"Step {i+1}: {s}")

    repair_prompt = (
        f"Question: {question}\n\nFill in masked steps:\n\n"
        + "\n".join(visible)
        + "\n\nComplete the solution. End with: The answer is <answer>.\n"
    )
    r_text, r_lps, r_top_lps, r_ptok, r_ctok = generate(model, tokenizer, repair_prompt, max_tokens, temperature=0.3, do_sample=True)
    answer = extract_answer(r_text) or extract_answer(text)
    return answer, ptok + ctok + r_ptok + r_ctok, ptok + r_ptok, ctok + r_ctok, f"FIRST:\n{text}\nREPAIR:\n{r_text}"

def run_backward_cloze(model, tokenizer, question, few_shot, max_tokens=512):
    prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step.\n"
    # Forward candidates
    answers = []
    total_ptok, total_ctok = 0, 0
    for _ in range(3):
        text, lps, top_lps, ptok, ctok = generate(model, tokenizer, prompt, max_tokens, temperature=0.7, do_sample=True)
        answers.append(extract_answer(text))
        total_ptok += ptok
        total_ctok += ctok
    anchor = Counter(answers).most_common(1)[0][0]

    # Backward verification
    bwd_prompt = (
        f"Question: {question}\n\nThe answer is: {anchor}\n\n"
        f"Working BACKWARD from this answer, verify it:\n"
        f"Final result: {anchor}\nPrevious step: [derive what leads to this]\n"
        f"After verification, state if {anchor} is correct. End with: The answer is <answer>.\n"
    )
    b_text, b_lps, b_top_lps, b_ptok, b_ctok = generate(model, tokenizer, bwd_prompt, max_tokens)
    b_answer = extract_answer(b_text)
    final = anchor if b_answer == anchor else b_answer
    return final, total_ptok + total_ctok + b_ptok + b_ctok, total_ptok + b_ptok, total_ctok + b_ctok, f"FWD answers: {answers}\nBWD: {b_text}"

def run_full_regeneration(model, tokenizer, question, few_shot, max_tokens=512):
    prompt = f"{few_shot}\nQuestion: {question}\nLet's think step by step.\n"
    text, lps, top_lps, ptok, ctok = generate(model, tokenizer, prompt, max_tokens)
    critique_prompt = (
        f"Question: {question}\n\nA student wrote:\n{text}\n\n"
        f"Review for errors and write a corrected solution. End with: The answer is <answer>.\n"
    )
    c_text, c_lps, c_top_lps, c_ptok, c_ctok = generate(model, tokenizer, critique_prompt, max_tokens, temperature=0.3, do_sample=True)
    answer = extract_answer(c_text) or extract_answer(text)
    return answer, ptok + ctok + c_ptok + c_ctok, ptok + c_ptok, ctok + c_ctok, f"FIRST:\n{text}\nREVISED:\n{c_text}"


STRATEGIES = {
    "standard_cot": run_standard_cot,
    "self_consistency": run_self_consistency,
    "compute_matched_sc": run_compute_matched_sc,
    "targeted_repair": run_targeted_repair,
    "random_repair": run_random_repair,
    "backward_cloze": run_backward_cloze,
    "full_regeneration": run_full_regeneration,
}


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model, tokenizer = load_model(model_name)

    benchmarks_config = [
        ("gsm8k", None),
        ("math", 500),
        ("strategyqa", None),
        ("arc_challenge", None),
    ]
    strategy_names = list(STRATEGIES.keys())
    seeds = [42]

    for bname, max_ex in benchmarks_config:
        log.info(f"\n{'='*60}\nBenchmark: {bname}\n{'='*60}")
        examples = load_benchmark(bname, max_examples=max_ex)
        few_shot = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")
        log.info(f"Loaded {len(examples)} examples")

        bm_dir = os.path.join(OUTPUT, bname)
        os.makedirs(bm_dir, exist_ok=True)

        for sname in strategy_names:
            ckpt = os.path.join(bm_dir, f".ckpt_{sname}.json")
            cached = []
            if os.path.exists(ckpt):
                with open(ckpt) as f:
                    cached = json.load(f)
            done_ids = {r["example_id"] for r in cached}
            remaining = [ex for ex in examples if ex.example_id not in done_ids]

            if not remaining:
                log.info(f"  {sname}: already complete ({len(cached)} results)")
                continue

            log.info(f"  {sname}: {len(cached)} done, {len(remaining)} remaining")
            strategy_fn = STRATEGIES[sname]
            results = list(cached)

            torch.manual_seed(42)
            for i, ex in enumerate(remaining):
                t0 = time.perf_counter()
                try:
                    answer, total_tok, p_tok, c_tok, trace = strategy_fn(
                        model, tokenizer, ex.question, few_shot, max_tokens=512,
                    )
                    elapsed = (time.perf_counter() - t0) * 1000
                    correct = check_answer(answer, ex.answer, ex.answer_type)
                    results.append({
                        "example_id": ex.example_id,
                        "prediction": answer,
                        "ground_truth": ex.answer,
                        "correct": correct,
                        "total_tokens": total_tok,
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "strategy": sname,
                        "elapsed_ms": elapsed,
                    })
                except Exception as e:
                    log.warning(f"  Error {sname}/{ex.example_id}: {e}")
                    results.append({
                        "example_id": ex.example_id, "correct": False,
                        "total_tokens": 0, "error": str(e),
                    })

                if (i + 1) % 50 == 0:
                    acc = sum(r.get("correct", False) for r in results) / len(results)
                    log.info(f"    {sname}: {len(results)}/{len(examples)} acc={acc:.4f}")
                    with open(ckpt, "w") as f:
                        json.dump(results, f, default=str)

            # Save final
            with open(ckpt, "w") as f:
                json.dump(results, f, default=str)

            acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
            tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens", 0) > 0]
            log.info(f"  {sname}: acc={acc:.4f} tokens={np.mean(tokens):.0f}")

        # Aggregate and save
        aggregate = {}
        all_results = {}
        for sname in strategy_names:
            ckpt = os.path.join(bm_dir, f".ckpt_{sname}.json")
            if os.path.exists(ckpt):
                with open(ckpt) as f:
                    results = json.load(f)
                all_results[sname] = results
                acc = sum(r.get("correct", False) for r in results) / max(len(results), 1)
                tokens = [r.get("total_tokens", 0) for r in results if r.get("total_tokens", 0) > 0]
                aggregate[sname] = {
                    "accuracy": acc,
                    "mean_tokens": float(np.mean(tokens)) if tokens else 0,
                    "n": len(results),
                }

        with open(os.path.join(bm_dir, "strategies.json"), "w") as f:
            json.dump({
                "benchmark": bname, "model": model_name,
                "aggregate": aggregate, "strategies": strategy_names,
            }, f, indent=2, default=str)

        # Print summary
        log.info(f"\n--- {bname} Summary ---")
        for sname in strategy_names:
            if sname in aggregate:
                log.info(f"  {sname}: {aggregate[sname]['accuracy']:.4f} ({aggregate[sname]['mean_tokens']:.0f} tokens)")

        # Clean checkpoints
        for sname in strategy_names:
            ckpt = os.path.join(bm_dir, f".ckpt_{sname}.json")
            if os.path.exists(ckpt):
                os.unlink(ckpt)

    log.info("\n=== DONE ===")


if __name__ == "__main__":
    main()
