#!/usr/bin/env python3
"""Replay historical AGD/PCS-style result JSONs through strict checking.

GPT-5.5 Pro Task 2 follow-up: the legacy `evaluation.check_answer_legacy_unsafe`
contains a substring fallback that contaminates answer comparison. This script
re-scores rows from a result file using `answer_extraction.check_answer_strict`
and emits a comparison table of legacy vs strict accuracy.

Usage:
    python code/replay_results_strict.py \
        --input results/agd/Qwen3.5-27B/agd_results.json \
        --out   results/agd/Qwen3.5-27B/agd_results_strict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_extraction import check_answer_strict
from evaluation import check_answer_legacy_unsafe


def _infer_answer_type(benchmark: str) -> str:
    return {
        "gsm8k": "numeric", "math_hard": "math_expression",
        "arc_challenge": "multiple_choice", "bbh_logic": "multiple_choice",
        "strategyqa": "boolean",
    }.get(benchmark, "text")


def replay_agd(data: dict) -> dict:
    """AGD result format: data["cells"][benchmark]["rows"] = list[per-example]."""
    out = {"benchmarks": {}, "summary": {}}
    cells = data.get("cells", {})
    for bench, cell in cells.items():
        atype = _infer_answer_type(bench)
        rows = cell.get("rows", [])
        legacy_correct = strict_correct = 0
        thr_keys = ["0.5", "0.75", "1.0"]
        thr_legacy = defaultdict(int)
        thr_strict = defaultdict(int)
        for row in rows:
            gt = row.get("ground_truth", "")
            sc8_correct = bool(row.get("sc8_correct", False))
            base_majority = row.get("base_majority", "")
            # SC(8) — re-score
            legacy_correct += int(check_answer_legacy_unsafe(base_majority, gt, atype))
            strict_correct += int(check_answer_strict(base_majority, gt, atype))
            # AGD per threshold
            for k in thr_keys:
                agd_pred = (row.get("agd", {}) or {}).get(k, {}).get("pred", "")
                thr_legacy[k] += int(check_answer_legacy_unsafe(agd_pred, gt, atype))
                thr_strict[k] += int(check_answer_strict(agd_pred, gt, atype))
        n = len(rows) or 1
        out["benchmarks"][bench] = {
            "n": len(rows),
            "answer_type": atype,
            "sc8_legacy_acc": legacy_correct / n,
            "sc8_strict_acc": strict_correct / n,
            "agd_legacy_acc": {k: thr_legacy[k] / n for k in thr_keys},
            "agd_strict_acc": {k: thr_strict[k] / n for k in thr_keys},
            "delta_sc8": (strict_correct - legacy_correct) / n,
        }
    # Aggregate
    bench_list = list(out["benchmarks"].values())
    if bench_list:
        out["summary"] = {
            "n_benchmarks": len(bench_list),
            "mean_sc8_legacy": sum(b["sc8_legacy_acc"] for b in bench_list) / len(bench_list),
            "mean_sc8_strict": sum(b["sc8_strict_acc"] for b in bench_list) / len(bench_list),
            "mean_delta_sc8": sum(b["delta_sc8"] for b in bench_list) / len(bench_list),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to AGD-style result JSON")
    p.add_argument("--out", default="", help="output path for strict-replay JSON")
    args = p.parse_args()

    data = json.load(open(args.input))
    if "cells" in data:
        result = replay_agd(data)
    else:
        raise SystemExit(
            f"Unrecognized format: {args.input}. Expected AGD-style with `cells`."
        )

    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
