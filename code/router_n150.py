#!/usr/bin/env python3
"""Question-Conditioned Strategy Router — N=150 within-cell + across-cell.

Uses the N=150 validation data (3 good cells) for within-cell 5-fold CV.
Adds per-problem per-strategy tokens (real cost) to evaluate compute-matched
comparison vs SC.
"""
from __future__ import annotations
import json, re, os
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

STRATEGIES = ["standard_cot", "self_consistency", "backward_cloze"]


def load_cell(path):
    data = json.load(open(path))
    # sweep_results.json format: {"model":..., "cells": {bench: {...}}}
    return data


def text_features(q):
    ql = q.lower(); words = q.split()
    n_digits = sum(c.isdigit() for c in q)
    return {
        "q_len": len(q), "n_words": len(words),
        "n_digits": n_digits, "n_numbers": len(re.findall(r"\d+", q)),
        "has_math_op": int(bool(re.search(r"[\+\-\*\/=\^]", q))),
        "has_dollar": int("$" in q), "has_percent": int("%" in q),
        "is_yn": int(any(q.strip().lower().startswith(w) for w in
                         ("is ", "are ", "can ", "do ", "does ", "did ",
                          "will ", "would ", "should ", "could "))),
        "is_choice": int(bool(re.search(r"\([a-eA-E]\)", q))),
        "has_calc": int(any(w in ql for w in ("calculate", "compute",
                                              "total", "how many", "how much"))),
        "has_why": int("why" in ql), "has_what": int("what" in ql),
        "has_if": int("if " in ql),
        "has_logic": int(any(w in ql for w in ("logic", "deduce",
                                               "premise", "follow"))),
    }


def assign_best(labels):
    correct = [s for s in STRATEGIES if labels[s]]
    if not correct:
        return "self_consistency"  # fallback: use SC by default
    if len(correct) == 3:
        return "standard_cot"  # cheapest when all agree
    # Prefer cheaper correct
    cost = {"standard_cot": 1, "backward_cloze": 4, "self_consistency": 8}
    return min(correct, key=lambda s: cost[s])


def build_rows(cell, model, benchmark):
    rows_by_s = cell["per_strategy_rows"]
    n = cell["n_examples"]
    rs = []
    for i in range(n):
        gt = rows_by_s["standard_cot"][i].get("ground_truth", "")
        labels = {s: int(bool(rows_by_s[s][i].get("correct", False))) for s in STRATEGIES}
        # per-example tokens
        tokens = {s: rows_by_s[s][i].get("total_tokens", 0) for s in STRATEGIES}
        # question reconstructed later
        rs.append({
            "model": model, "benchmark": benchmark,
            "example_id": rows_by_s["standard_cot"][i].get("example_id", f"{benchmark}_{i}"),
            "gt": gt, "labels": labels, "tokens": tokens,
            "best": assign_best(labels),
            "any_correct": int(any(labels.values())),
        })
    return rs


def load_questions_via_datasets(rows):
    """Try to load question text via datasets (local cache)."""
    os.environ["HF_HUB_OFFLINE"] = ""  # allow online via mirror
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    q = {}
    benchmarks = sorted({r["benchmark"] for r in rows})
    try:
        import sys; sys.path.insert(0, "code")
        from benchmarks import load_benchmark, load_math, load_bbh
        for b in benchmarks:
            try:
                if b == "math_hard":
                    items = load_math(levels=[4, 5])
                elif b == "bbh_logic":
                    items = load_bbh(subtasks=["logical_deduction_five_objects"])
                else:
                    items = load_benchmark(b)
                for it in items:
                    q[it.example_id] = it.question
                print(f"  {b}: got {sum(1 for r in rows if r['benchmark']==b and r['example_id'] in q)}/"
                      f"{sum(1 for r in rows if r['benchmark']==b)}")
            except Exception as e:
                print(f"  {b}: {e}")
    except Exception as e:
        print(f"Can't import benchmarks: {e}")
    return q


def eval_router(preds, test, strat_labels):
    correct = 0
    total_tokens = 0
    route_counter = Counter()
    for j, r in enumerate(test):
        s = strat_labels[int(preds[j])]
        route_counter[s] += 1
        correct += r["labels"][s]
        total_tokens += r["tokens"].get(s, 0)
    return {
        "acc": correct / max(len(test), 1),
        "mean_tokens": total_tokens / max(len(test), 1),
        "routes": dict(route_counter),
    }


def main():
    # Load all good cells (meta-sweep N=30 + N=150 validation)
    cells = []
    # Meta-sweep N=30
    for d in sorted(Path("results/meta").iterdir()):
        if not d.is_dir(): continue
        f = d / "sweep_results.json"
        if f.exists():
            data = json.load(open(f))
            for b, c in data.get("cells", {}).items():
                if c.get("oracle_sc_gap", 0) >= 0.15:
                    cells.append({"model": d.name, "benchmark": b, "cell": c, "N": 30})
    # N=150 validation
    n150_map = {
        "1.5B_strategyqa": ("Qwen2.5-1.5B", "strategyqa"),
        "7B_strategyqa": ("Qwen2.5-7B", "strategyqa"),
        "3B_bbh_logic": ("Qwen2.5-3B", "bbh_logic"),
    }
    for tag, (m, b) in n150_map.items():
        f = Path(f"results/meta_n150/{tag}/sweep_results.json")
        if f.exists():
            data = json.load(open(f))
            for bench, c in data.get("cells", {}).items():
                cells.append({"model": m, "benchmark": bench,
                              "cell": c, "N": c["n_examples"]})

    print(f"Loaded {len(cells)} cells")

    # Build rows
    all_rows = []
    for ci, cd in enumerate(cells):
        rs = build_rows(cd["cell"], cd["model"], cd["benchmark"])
        for r in rs:
            r["cell_tag"] = f"{cd['model']}/{cd['benchmark']}/N{cd['N']}"
        all_rows.extend(rs)
    print(f"Total rows: {len(all_rows)}")

    # Load questions
    q = load_questions_via_datasets(all_rows)
    for r in all_rows:
        r["question"] = q.get(r["example_id"], r["example_id"])
    resolved = sum(1 for r in all_rows if r["question"] != r["example_id"])
    print(f"Resolved: {resolved}/{len(all_rows)}")

    # Only keep rows where we have real question text
    data_rows = [r for r in all_rows if r["question"] != r["example_id"]]
    for r in data_rows:
        r["feats"] = text_features(r["question"])
    print(f"Keeping {len(data_rows)} rows with real question text")

    feature_names = list(data_rows[0]["feats"].keys())
    strat_labels = STRATEGIES  # no "none" class (we default to SC)
    label_to_idx = {s: i for i, s in enumerate(strat_labels)}

    # === Within-cell 5-fold CV on N=150 cells ===
    print(f"\n{'='*70}")
    print("Within-cell 5-fold CV (N=150 validation cells only)")
    print(f"{'='*70}")
    print(f"{'Cell':<30} {'SC':>6} {'Oracle':>7} {'Rt':>6} {'Cap':>6} {'TokRt':>6}")

    within_summary = []
    for cell_tag in {r["cell_tag"] for r in data_rows if "N150" in r["cell_tag"]}:
        cell_rows = [r for r in data_rows if r["cell_tag"] == cell_tag]
        if len(cell_rows) < 20:
            continue
        X_all = np.array([[r["feats"][f] for f in feature_names] for r in cell_rows])
        y_all = np.array([label_to_idx[r["best"]] for r in cell_rows])
        q_all = [r["question"] for r in cell_rows]

        preds = np.zeros(len(cell_rows), dtype=int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            splits = list(skf.split(X_all, y_all))
        except ValueError:
            splits = [(np.arange(len(X_all)), np.arange(0))]

        for tr_idx, te_idx in splits:
            scaler = StandardScaler().fit(X_all[tr_idx])
            Xtr = scaler.transform(X_all[tr_idx])
            Xte = scaler.transform(X_all[te_idx])
            try:
                tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2),
                                         min_df=2, stop_words="english").fit(
                    [q_all[i] for i in tr_idx])
                Xtr = hstack([csr_matrix(Xtr), tfidf.transform([q_all[i] for i in tr_idx])])
                Xte = hstack([csr_matrix(Xte), tfidf.transform([q_all[i] for i in te_idx])])
            except Exception:
                pass
            clf = LogisticRegression(max_iter=5000, C=1.0,
                                     class_weight="balanced",
                                     solver="saga").fit(Xtr, y_all[tr_idx])
            preds[te_idx] = clf.predict(Xte)

        res = eval_router(preds, cell_rows, strat_labels)
        sc_acc = sum(r["labels"]["self_consistency"] for r in cell_rows) / len(cell_rows)
        oracle_acc = sum(r["any_correct"] for r in cell_rows) / len(cell_rows)
        sc_tok = sum(r["tokens"]["self_consistency"] for r in cell_rows) / len(cell_rows)
        gap = oracle_acc - sc_acc
        cap = (res["acc"] - sc_acc) / gap if gap > 0 else 0
        tok_ratio = res["mean_tokens"] / sc_tok if sc_tok > 0 else 1.0

        print(f"{cell_tag:<30} {sc_acc:>6.3f} {oracle_acc:>7.3f} {res['acc']:>6.3f} "
              f"{cap:>+6.2f} {tok_ratio:>6.2f}x")
        within_summary.append({"cell": cell_tag, "sc": sc_acc, "oracle": oracle_acc,
                              "router_acc": res["acc"], "gap_capture": cap,
                              "token_ratio": tok_ratio, "routes": res["routes"]})

    avg_cap = np.mean([s["gap_capture"] for s in within_summary]) if within_summary else 0
    avg_tok = np.mean([s["token_ratio"] for s in within_summary]) if within_summary else 1
    print(f"\n{'Within-cell average':<30} {'':<6} {'':<7} {'':<6} "
          f"{avg_cap:>+6.2f} {avg_tok:>6.2f}x")

    # === Decision ===
    print(f"\n{'='*70}")
    print("DECISION")
    print(f"{'='*70}")
    if avg_cap >= 0.30 and avg_tok < 0.5:
        verdict = "POSITIVE"
        print(f">>> POSITIVE: router captures {avg_cap:.0%} at {avg_tok:.0%} SC compute")
    elif avg_cap >= 0.15:
        verdict = "MARGINAL"
        print(f">>> MARGINAL: {avg_cap:.0%} capture / {avg_tok:.0%} compute")
    else:
        verdict = "NEGATIVE"
        print(f">>> NEGATIVE: only {avg_cap:.0%} — router isn't extracting signal")

    Path("results/router_n150").mkdir(parents=True, exist_ok=True)
    with open("results/router_n150/summary.json", "w") as f:
        json.dump({"within_summary": within_summary, "avg_capture": avg_cap,
                  "avg_token_ratio": avg_tok, "verdict": verdict},
                 f, indent=2, default=str)
    print(f"Saved: results/router_n150/summary.json")


if __name__ == "__main__":
    main()
