#!/usr/bin/env python3
"""Analyze BAV experiment: token-accuracy Pareto vs baselines.

Combines BAV results with the original pilot's baselines to compute:
  - token-accuracy Pareto front
  - BAV agreement rate (when forward == backward)
  - Conditional accuracies (agreed vs disagreed problems)
  - Statistical test of BAV vs SC at matched token budgets

Usage:
    python3 code/analyze_bav.py \\
        --bav results/bav/pilot_results.json \\
        --pilot results/pilot/pilot_results.json \\
        --output results/bav
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load(p):
    with open(p) as f:
        return json.load(f)


def strategy_stats(data, sname):
    res = data["strategy_results"].get(sname, {})
    correct = sum(1 for v in res.values() if v.get("correct"))
    tokens = [v.get("total_tokens", 0) for v in res.values() if v.get("total_tokens", 0) > 0]
    n = len(res)
    return {
        "name": sname,
        "n": n,
        "correct": correct,
        "accuracy": correct / max(n, 1),
        "mean_tokens": float(np.mean(tokens)) if tokens else 0.0,
        "median_tokens": float(np.median(tokens)) if tokens else 0.0,
    }


def bav_diagnostics(bav_data, sname="bav"):
    """Analyze BAV internal metrics: agreement rate, agreed/disagreed acc."""
    res = bav_data["strategy_results"].get(sname, {})
    n = len(res)
    n_agreed = 0
    n_disagreed = 0
    correct_agreed = 0
    correct_disagreed = 0
    for v in res.values():
        meta = v.get("step_metadata", [{}])
        if meta and meta[0].get("agreed"):
            n_agreed += 1
            if v.get("correct"):
                correct_agreed += 1
        else:
            n_disagreed += 1
            if v.get("correct"):
                correct_disagreed += 1
    return {
        "n_total": n,
        "n_agreed": n_agreed,
        "n_disagreed": n_disagreed,
        "agreement_rate": n_agreed / max(n, 1),
        "acc_agreed": correct_agreed / max(n_agreed, 1) if n_agreed > 0 else 0.0,
        "acc_disagreed": correct_disagreed / max(n_disagreed, 1) if n_disagreed > 0 else 0.0,
        "overall_acc": (correct_agreed + correct_disagreed) / max(n, 1),
    }


def pareto_front(points):
    """Compute Pareto front: lowest tokens for each accuracy level."""
    # Points: list of (tokens, acc, name). Pareto: no other point has lower tokens AND higher acc.
    pts = sorted(points, key=lambda x: (x[0], -x[1]))
    front = []
    best_acc = -1
    for tokens, acc, name in pts:
        if acc > best_acc:
            front.append((tokens, acc, name))
            best_acc = acc
    return front


def paired_bootstrap(a_correct, b_correct, n_boot=10000, seed=42):
    """Paired bootstrap for A - B accuracy difference."""
    rng = np.random.default_rng(seed)
    a = np.array(a_correct, dtype=float)
    b = np.array(b_correct, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    observed = float(np.mean(a) - np.mean(b))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(a[idx]) - np.mean(b[idx])
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    centered = diffs - np.mean(diffs)
    p = float(np.mean(np.abs(centered) >= abs(observed)))
    return {"delta": observed, "ci_95": (ci_lo, ci_hi), "p_value": p}


def aligned_results(bav_data, pilot_data, sname_bav, sname_pilot):
    """Return aligned correctness lists on common example_ids."""
    bav_res = bav_data["strategy_results"].get(sname_bav, {})
    pilot_res = pilot_data["strategy_results"].get(sname_pilot, {})
    common = sorted(set(bav_res.keys()) & set(pilot_res.keys()))
    a = [1 if bav_res[k].get("correct") else 0 for k in common]
    b = [1 if pilot_res[k].get("correct") else 0 for k in common]
    return a, b, common


def plot_pareto(stats_list, bav_stat, output_path):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(9, 6))

    baselines = [s for s in stats_list if not s["name"].startswith("bav")]
    bavs = [s for s in stats_list if s["name"].startswith("bav") or s["name"] == "bav"]

    # Scatter all strategies
    for s in baselines:
        ax.scatter(s["mean_tokens"], s["accuracy"] * 100,
                   s=80, alpha=0.7, edgecolor="black", linewidth=0.5,
                   color="#90A4AE", label="_baseline")
        ax.annotate(s["name"].replace("_", " "), (s["mean_tokens"], s["accuracy"] * 100),
                    fontsize=7, textcoords="offset points", xytext=(5, 3))

    for s in bavs:
        ax.scatter(s["mean_tokens"], s["accuracy"] * 100,
                   s=140, alpha=0.9, edgecolor="black", linewidth=1.0,
                   color="#E53935", marker="*", label="BAV")
        ax.annotate("BAV", (s["mean_tokens"], s["accuracy"] * 100),
                    fontsize=9, fontweight="bold",
                    textcoords="offset points", xytext=(8, 5))

    # Pareto front
    points = [(s["mean_tokens"], s["accuracy"], s["name"]) for s in stats_list]
    front = pareto_front(points)
    if len(front) > 1:
        xs = [p[0] for p in front]
        ys = [p[1] * 100 for p in front]
        ax.plot(xs, ys, "--", color="#2E7D32", alpha=0.6, linewidth=1.5,
                label="Pareto front")

    ax.set_xlabel("Mean tokens per problem", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Token-Accuracy Pareto: BAV vs Baselines (GSM8K, Qwen3.5-27B)",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bav", required=True, help="BAV pilot_results.json")
    parser.add_argument("--pilot", required=True, help="Original pilot_results.json")
    parser.add_argument("--output", default="results/bav", help="Output dir")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    bav_data = load(args.bav)
    pilot_data = load(args.pilot)

    # Collect stats
    print("=" * 70)
    print("Per-strategy stats")
    print("=" * 70)

    strategies_of_interest = [
        ("standard_cot", pilot_data),
        ("compute_matched_sc", pilot_data),
        ("sc_k3", bav_data),
        ("sc_k5", bav_data),
        ("self_consistency", pilot_data),
        ("backward_cloze", pilot_data),
        ("bav", bav_data),
    ]

    stats_list = []
    for sname, data_src in strategies_of_interest:
        s = strategy_stats(data_src, sname)
        stats_list.append(s)
        print(f"  {s['name']:25s}  acc={s['accuracy']:.1%}  "
              f"tokens={s['mean_tokens']:6.0f}  n={s['n']}")

    # BAV diagnostics
    print("\n" + "=" * 70)
    print("BAV diagnostics")
    print("=" * 70)
    bav_diag = bav_diagnostics(bav_data)
    print(f"  Agreement rate:    {bav_diag['agreement_rate']:.1%}  "
          f"({bav_diag['n_agreed']}/{bav_diag['n_total']})")
    print(f"  Acc when agreed:   {bav_diag['acc_agreed']:.1%}")
    print(f"  Acc when disagreed:{bav_diag['acc_disagreed']:.1%}")
    print(f"  Overall BAV acc:   {bav_diag['overall_acc']:.1%}")

    # Statistical comparisons: BAV vs SC at matched budgets
    print("\n" + "=" * 70)
    print("BAV vs budget-matched SC (paired bootstrap)")
    print("=" * 70)

    for target in ["sc_k3", "sc_k5", "self_consistency"]:
        a, b, ids = aligned_results(bav_data, pilot_data if target == "self_consistency"
                                    else bav_data, "bav", target)
        if not ids:
            # Maybe cross-file
            a, b, ids = aligned_results(bav_data, pilot_data, "bav", target)
        if not ids:
            print(f"  BAV vs {target}: no common examples, skip")
            continue
        test = paired_bootstrap(a, b)
        bav_acc = sum(a) / len(a)
        tgt_acc = sum(b) / len(b)
        print(f"  BAV ({bav_acc:.1%}) vs {target} ({tgt_acc:.1%}):  "
              f"delta={test['delta']:+.3f}  "
              f"CI95=[{test['ci_95'][0]:+.3f},{test['ci_95'][1]:+.3f}]  "
              f"p={test['p_value']:.3f}")

    # Pareto plot
    print("\n" + "=" * 70)
    print("Pareto analysis")
    print("=" * 70)
    points = [(s["mean_tokens"], s["accuracy"], s["name"]) for s in stats_list
              if s["mean_tokens"] > 0]
    front = pareto_front(points)
    print("  Pareto-optimal strategies (by tokens ↑):")
    for p in front:
        tag = "  <-- BAV" if p[2] == "bav" else ""
        print(f"    {p[2]:25s}  tokens={p[0]:6.0f}  acc={p[1]:.1%}{tag}")

    plot_pareto(stats_list, None, os.path.join(args.output, "bav_pareto.pdf"))

    # Save full analysis
    out = {
        "stats": stats_list,
        "bav_diagnostics": bav_diag,
        "pareto_front": [
            {"strategy": p[2], "tokens": p[0], "accuracy": p[1]} for p in front
        ],
    }
    with open(os.path.join(args.output, "bav_analysis.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {args.output}/bav_analysis.json")


if __name__ == "__main__":
    main()
