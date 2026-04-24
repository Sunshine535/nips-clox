#!/usr/bin/env python3
"""CLOX-PCS post-hoc analysis — paired bootstrap, McNemar, accuracy/token tables.

GPT-5.5 Pro Task 9 companion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import paired_bootstrap_ci


def mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided McNemar exact-ish (binomial) test on discordant pairs.

    b = count(A wrong, B correct), c = count(A correct, B wrong).
    """
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    # If B better than A (expected under H1), test b >= observed
    p = binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    return float(p)


def analyze(eval_json_path: str, compare: list[str]):
    data = json.load(open(eval_json_path))
    rows = data["per_example"]
    summary = data["summary"]

    arms = compare
    arm_correct = {a: np.array([1 if r["arms"][a]["correct"] else 0 for r in rows])
                   for a in arms}

    out = {"n_examples": len(rows), "arms": summary["arms"], "paired": {}}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            d = arm_correct[b] - arm_correct[a]
            ci = paired_bootstrap_ci(arm_correct[b], arm_correct[a],
                                     num_samples=1000, seed=11)
            # McNemar b=count(a wrong, b correct), c=count(a correct, b wrong)
            b_only = int(((arm_correct[a] == 0) & (arm_correct[b] == 1)).sum())
            a_only = int(((arm_correct[a] == 1) & (arm_correct[b] == 0)).sum())
            p = mcnemar_pvalue(b_only, a_only)
            out["paired"][f"{b}_minus_{a}"] = {
                "mean_diff_acc": float(d.mean()),
                "bootstrap_ci_2.5%": ci[0],
                "bootstrap_ci_97.5%": ci[1],
                f"{a}_only_correct": a_only,
                f"{b}_only_correct": b_only,
                "mcnemar_p_value": p,
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, required=True, help="eval.json path")
    p.add_argument("--compare", type=str, default="A,B,C")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    arms = [x.strip() for x in args.compare.split(",")]
    result = analyze(args.runs, arms)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
