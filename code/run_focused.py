#!/usr/bin/env python3
"""Focused experiment: 200 GSM8K examples, all strategies, single seed.
Gets the complete strategy comparison fast for paper drafting.
"""
import sys, os, json, time, logging, math, re
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmarks import load_gsm8k, load_math, load_strategyqa, load_arc_challenge, FEW_SHOT_PROMPTS
from evaluation import check_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clox_focused")

OUTPUT = "/home/claude/nips-clox/results/v2_focused"


def load_model(name):
    log.info(f"Loading {name}...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    return model, tok


def gen(model, tok, prompt, max_tok=512, temp=0.0, sample=False):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=3072)
    inp = ids["input_ids"].to(model.device)
    plen = inp.shape[1]
    kw = {"max_new_tokens": max_tok, "do_sample": sample, "pad_token_id": tok.pad_token_id or tok.eos_token_id,
          "return_dict_in_generate": True, "output_scores": True}
    if sample and temp > 0:
        kw["temperature"] = temp
    with torch.inference_mode():
        out = model.generate(inp, **kw)
    gids = out.sequences[0][plen:]
    text = tok.decode(gids, skip_special_tokens=True)
    lps, tlps = [], []
    if out.scores:
        for i, sc in enumerate(out.scores):
            if i >= len(gids): break
            lp = F.log_softmax(sc[0].float(), dim=-1)
            lps.append(float(lp[gids[i]].cpu()))
            tv, ti = torch.topk(lp, 20)
            tlps.append({int(t): float(v) for t, v in zip(ti.cpu(), tv.cpu())})
    return text, lps, tlps, plen, len(gids)


def extract_ans(text):
    for pat in [r"\\boxed\{([^}]+)\}", r"####\s*(.+)", r"(?:the answer is|answer:)\s*[:\s]*([^\n.;]+)",
                r"(?:Therefore|So|Thus),?\s*(?:the answer is)?\s*([^\n.;]+)"]:
        ms = list(re.finditer(pat, text, re.IGNORECASE))
        if ms:
            a = ms[-1].group(1).strip()
            if 0 < len(a) < 200: return a
    for l in reversed([l.strip() for l in text.strip().split("\n") if l.strip()]):
        n = re.search(r"[-−]?\d+(?:\.\d+)?", l)
        if n: return n.group()
        if len(l) < 50: return l
    return ""


def steps(text):
    ps = re.split(r'\n(?=\d+[\.\):])', text)
    if len(ps) >= 3: return [s.strip() for s in ps if s.strip() and len(s.split()) >= 3]
    ps = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in ps if s.strip() and len(s.split()) >= 3]


def step_ent(tlps, nsteps, ntok):
    if not nsteps or not tlps: return []
    tps = max(ntok // nsteps, 1)
    es = []
    for i in range(nsteps):
        s, e = i*tps, min((i+1)*tps, len(tlps))
        se = []
        for p in range(s, e):
            if p < len(tlps):
                probs = [math.exp(v) for v in tlps[p].values()]
                t = sum(probs)
                if t > 0:
                    probs = [p/t for p in probs]
                    se.append(-sum(p*math.log(max(p,1e-30)) for p in probs if p>0))
        es.append(float(np.mean(se)) if se else 0.0)
    return es


# Strategies
def s_cot(m, t, q, fs, mt=512):
    text, lps, tlps, pt, ct = gen(m, t, f"{fs}\nQuestion: {q}\nLet's think step by step.\n", mt)
    return extract_ans(text), pt+ct, ct, "standard_cot"

def s_sc(m, t, q, fs, k=5, mt=512):
    prompt = f"{fs}\nQuestion: {q}\nLet's think step by step.\n"
    ans, tpt, tct = [], 0, 0
    for _ in range(k):
        text, _, _, pt, ct = gen(m, t, prompt, mt, temp=0.7, sample=True)
        ans.append(extract_ans(text)); tpt += pt; tct += ct
    return Counter(ans).most_common(1)[0][0], tpt+tct, tct, "self_consistency"

def s_sc2(m, t, q, fs, mt=512):
    return s_sc(m, t, q, fs, k=2, mt=mt)

def s_repair(m, t, q, fs, mt=512, targeted=True):
    prompt = f"{fs}\nQuestion: {q}\nLet's think step by step.\n"
    text, lps, tlps, pt, ct = gen(m, t, prompt, mt)
    st = steps(text)
    if len(st) < 2: return extract_ans(text), pt+ct, ct, "targeted_repair" if targeted else "random_repair"

    if targeted:
        ents = step_ent(tlps, len(st), len(lps))
        nm = max(1, int(len(st)*0.4))
        ranked = sorted(range(len(ents)), key=lambda i: ents[i], reverse=True)
        masked = sorted(ranked[:nm])
        tag = "[HIGH UNCERTAINTY - CORRECT THIS]"
    else:
        rng = np.random.default_rng(hash(q)%(2**32)+12345)
        nm = max(1, int(len(st)*0.4))
        masked = sorted(rng.choice(len(st), min(nm, len(st)), replace=False).tolist())
        tag = "[MASKED - fill in]"

    vis = [f"Step {i+1}: {tag}" if i in masked else f"Step {i+1}: {s}" for i, s in enumerate(st)]
    rp = f"Question: {q}\n\nFix the marked steps:\n\n" + "\n".join(vis) + "\n\nEnd with: The answer is <answer>.\n"
    rt, _, _, rpt, rct = gen(m, t, rp, mt, temp=0.3, sample=True)
    a = extract_ans(rt) or extract_ans(text)
    return a, pt+ct+rpt+rct, ct+rct, "targeted_repair" if targeted else "random_repair"

def s_targeted(m, t, q, fs, mt=512): return s_repair(m, t, q, fs, mt, targeted=True)
def s_random(m, t, q, fs, mt=512): return s_repair(m, t, q, fs, mt, targeted=False)

def s_backward(m, t, q, fs, mt=512):
    prompt = f"{fs}\nQuestion: {q}\nLet's think step by step.\n"
    ans, tpt, tct = [], 0, 0
    for _ in range(3):
        text, _, _, pt, ct = gen(m, t, prompt, mt, temp=0.7, sample=True)
        ans.append(extract_ans(text)); tpt += pt; tct += ct
    anchor = Counter(ans).most_common(1)[0][0]
    bp = f"Question: {q}\n\nThe answer is: {anchor}\nWorking BACKWARD, verify this. End with: The answer is <answer>.\n"
    bt, _, _, bpt, bct = gen(m, t, bp, mt)
    ba = extract_ans(bt)
    final = anchor if ba == anchor else ba
    return final, tpt+tct+bpt+bct, tct+bct, "backward_cloze"

def s_regen(m, t, q, fs, mt=512):
    prompt = f"{fs}\nQuestion: {q}\nLet's think step by step.\n"
    text, _, _, pt, ct = gen(m, t, prompt, mt)
    cp = f"Question: {q}\n\nReview and correct:\n{text}\n\nEnd with: The answer is <answer>.\n"
    ct2, _, _, cpt, cct = gen(m, t, cp, mt, temp=0.3, sample=True)
    a = extract_ans(ct2) or extract_ans(text)
    return a, pt+ct+cpt+cct, ct+cct, "full_regeneration"

STRATS = [
    ("standard_cot", s_cot),
    ("self_consistency_5", lambda m,t,q,f,mt=512: s_sc(m,t,q,f,k=5,mt=mt)),
    ("compute_matched_sc", s_sc2),
    ("targeted_repair", s_targeted),
    ("random_repair", s_random),
    ("backward_cloze", s_backward),
    ("full_regeneration", s_regen),
]


def run_benchmark(model, tok, bname, examples, few_shot, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for sname, sfn in STRATS:
        log.info(f"  {sname}...")
        torch.manual_seed(42)
        res = []
        t0 = time.time()
        for i, ex in enumerate(examples):
            try:
                ans, ttok, ctok, _ = sfn(model, tok, ex.question, few_shot)
                correct = check_answer(ans, ex.answer, ex.answer_type)
                res.append({"id": ex.example_id, "pred": ans, "gt": ex.answer,
                           "correct": correct, "tokens": ttok, "comp_tokens": ctok})
            except Exception as e:
                res.append({"id": ex.example_id, "correct": False, "error": str(e)})
            if (i+1) % 50 == 0:
                acc = sum(r["correct"] for r in res)/len(res)
                log.info(f"    {sname}: {i+1}/{len(examples)} acc={acc:.4f}")

        elapsed = time.time() - t0
        acc = sum(r["correct"] for r in res)/max(len(res),1)
        toks = [r.get("tokens",0) for r in res if r.get("tokens",0)>0]
        log.info(f"  {sname}: acc={acc:.4f} tokens={np.mean(toks):.0f} time={elapsed:.0f}s")
        results[sname] = {"accuracy": acc, "mean_tokens": float(np.mean(toks)) if toks else 0,
                          "mean_comp_tokens": float(np.mean([r.get("comp_tokens",0) for r in res if r.get("comp_tokens",0)>0])) if toks else 0,
                          "n": len(res), "time_s": elapsed, "per_example": res}

    with open(os.path.join(output_dir, f"{bname}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\n--- {bname} Summary ---")
    log.info(f"{'Strategy':<25s} {'Acc':>8s} {'Tokens':>8s} {'CompTok':>8s} {'Time':>8s}")
    log.info("-" * 62)
    for sname, _ in STRATS:
        r = results[sname]
        log.info(f"{sname:<25s} {r['accuracy']:>8.4f} {r['mean_tokens']:>8.0f} {r['mean_comp_tokens']:>8.0f} {r['time_s']:>7.0f}s")
    return results


def main():
    model, tok = load_model("Qwen/Qwen2.5-7B-Instruct")

    benchmarks = [
        ("gsm8k", load_gsm8k(max_examples=200)),
        ("strategyqa", load_strategyqa(max_examples=200)),
        ("arc_challenge", load_arc_challenge(max_examples=200)),
    ]

    all_results = {}
    for bname, examples in benchmarks:
        log.info(f"\n{'='*60}\n{bname} ({len(examples)} examples)\n{'='*60}")
        fs = FEW_SHOT_PROMPTS.get(bname.split("_")[0], "")
        bdir = os.path.join(OUTPUT, bname)

        # Skip if already done
        out_file = os.path.join(bdir, f"{bname}.json")
        if os.path.exists(out_file):
            log.info(f"  Already done, loading...")
            with open(out_file) as f:
                all_results[bname] = json.load(f)
            continue

        all_results[bname] = run_benchmark(model, tok, bname, examples, fs, bdir)

    # Final summary across all benchmarks
    log.info(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    header = f"{'Strategy':<25s}"
    for bname, _ in benchmarks:
        header += f" {bname:>12s}"
    log.info(header)
    log.info("-" * (25 + 13*len(benchmarks)))
    for sname, _ in STRATS:
        line = f"{sname:<25s}"
        for bname, _ in benchmarks:
            r = all_results.get(bname, {}).get(sname, {})
            line += f" {r.get('accuracy',0):>12.4f}"
        log.info(line)

    # Save combined summary
    summary = {}
    for bname, _ in benchmarks:
        summary[bname] = {sname: {"accuracy": all_results.get(bname,{}).get(sname,{}).get("accuracy",0),
                                   "mean_tokens": all_results.get(bname,{}).get(sname,{}).get("mean_tokens",0)}
                          for sname, _ in STRATS}
    with open(os.path.join(OUTPUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nResults saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
