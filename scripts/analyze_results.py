#!/usr/bin/env python3
"""CLOX result analysis: topology metrics, crossover validation, statistical tests, tables."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
from evaluation import (
    bonferroni_correction,
    cohens_d,
    mcnemar_test,
    paired_bootstrap_ci,
)


def load_all_results(results_dir: str) -> dict:
    """Load all benchmark result JSON files from a model's results directory."""
    results = {}
    for f in Path(results_dir).glob("*_results.json"):
        benchmark = f.stem.replace("_results", "")
        with open(f) as fh:
            results[benchmark] = json.load(fh)
    return results


def compute_strategy_table(results: dict) -> str:
    """Generate a LaTeX table: benchmarks × strategies with accuracy ± std."""
    benchmarks = sorted(results.keys())
    strategies = None
    for b in benchmarks:
        strategies = results[b].get("strategy_names", [])
        if strategies:
            break
    if not strategies:
        return "No strategies found."

    header = "Strategy & " + " & ".join(b.replace("_", "\\_") for b in benchmarks) + " \\\\"
    rows = [header, "\\midrule"]

    best_per_benchmark = {}
    for b in benchmarks:
        agg = results[b].get("aggregate", {})
        best_acc = 0
        best_strat = ""
        for s in strategies:
            acc = agg.get(s, {}).get("mean_accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_strat = s
        best_per_benchmark[b] = best_strat

    for s in strategies:
        cells = []
        for b in benchmarks:
            agg = results[b].get("aggregate", {})
            info = agg.get(s, {})
            acc = info.get("mean_accuracy", 0) * 100
            std = info.get("std_accuracy", 0) * 100
            cell = f"{acc:.1f} \\scriptsize{{±{std:.1f}}}"
            if best_per_benchmark.get(b) == s:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        name = s.replace("_", " ").title()
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")

    return "\n".join(rows)


def compute_topology_summary(results: dict) -> dict:
    """Extract per-benchmark topology metrics from CLOX-Adaptive results."""
    topology_summary = {}
    for benchmark, data in results.items():
        per_strat = data.get("per_strategy_results", {})
        adaptive_results = per_strat.get("clox_adaptive", {})
        epls = []
        recoverabilities = []
        for seed_key, examples in adaptive_results.items():
            for ex in examples:
                topo = ex.get("topology", {})
                if topo.get("epl") is not None:
                    epls.append(topo["epl"])
                if topo.get("recoverability") is not None:
                    recoverabilities.append(topo["recoverability"])

        topology_summary[benchmark] = {
            "mean_epl": float(np.mean(epls)) if epls else None,
            "std_epl": float(np.std(epls)) if epls else None,
            "mean_recoverability": float(np.mean(recoverabilities)) if recoverabilities else None,
            "std_recoverability": float(np.std(recoverabilities)) if recoverabilities else None,
            "n_samples": len(epls),
        }
    return topology_summary


def validate_crossover(results: dict, topology: dict) -> dict:
    """Check if topology predicts the winning strategy per benchmark."""
    predictions = {}
    for benchmark, topo in topology.items():
        epl = topo.get("mean_epl")
        rbar = topo.get("mean_recoverability")
        if epl is None or rbar is None:
            predictions[benchmark] = {"predicted": "unknown", "actual": "unknown", "match": False}
            continue

        # Theory prediction:
        # Low EPL + High r̄ → masking strategies should win
        # High EPL + Low r̄ → self-consistency should win
        if epl < 3.0 and rbar > 0.5:
            predicted = "masking"
        elif epl > 5.0 or rbar < 0.3:
            predicted = "self_consistency"
        else:
            predicted = "adaptive"

        # Find actual winner
        agg = results[benchmark].get("aggregate", {})
        best_acc = 0
        best_strat = ""
        for s, info in agg.items():
            if isinstance(info, dict) and "mean_accuracy" in info:
                acc = info["mean_accuracy"]
                if acc > best_acc:
                    best_acc = acc
                    best_strat = s

        # Classify actual winner
        masking_strategies = {"uncertainty_masked_repair", "random_masked_repair", "hierarchical_repair", "backward_cloze"}
        if best_strat == "self_consistency":
            actual = "self_consistency"
        elif best_strat in masking_strategies:
            actual = "masking"
        elif best_strat == "clox_adaptive":
            actual = "adaptive"
        else:
            actual = "other"

        match = predicted == actual or (predicted == "adaptive" and actual in {"masking", "self_consistency"})

        predictions[benchmark] = {
            "epl": epl,
            "rbar": rbar,
            "predicted": predicted,
            "actual_winner": best_strat,
            "actual_category": actual,
            "best_accuracy": best_acc,
            "match": match,
        }
    return predictions


def compute_pairwise_stats(results: dict) -> dict:
    """Compute pairwise statistical tests between all strategies for each benchmark."""
    stats = {}
    for benchmark, data in results.items():
        agg = data.get("aggregate", {})
        pairwise = agg.get("pairwise_vs_baseline", {})
        stats[benchmark] = pairwise
    return stats


def generate_report(results_dir: str, output_path: str | None = None):
    """Generate full analysis report."""
    results = load_all_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"=== CLOX Analysis Report ({results_dir}) ===\n")

    # 1. Main results table
    print("## Main Results Table (LaTeX)\n")
    print(compute_strategy_table(results))
    print()

    # 2. Topology summary
    print("## Topology Summary\n")
    topology = compute_topology_summary(results)
    for b, t in topology.items():
        if t["mean_epl"] is not None:
            print(f"  {b}: EPL={t['mean_epl']:.2f}±{t['std_epl']:.2f}, "
                  f"r̄={t['mean_recoverability']:.3f}±{t['std_recoverability']:.3f} "
                  f"(n={t['n_samples']})")
        else:
            print(f"  {b}: no topology data")
    print()

    # 3. Crossover validation
    print("## Crossover Validation\n")
    crossover = validate_crossover(results, topology)
    matches = 0
    total = 0
    for b, c in crossover.items():
        status = "MATCH" if c["match"] else "MISMATCH"
        print(f"  {b}: predicted={c['predicted']}, actual={c.get('actual_category', '?')} "
              f"(winner={c.get('actual_winner', '?')}, acc={c.get('best_accuracy', 0):.3f}) → {status}")
        if c["predicted"] != "unknown":
            total += 1
            matches += int(c["match"])
    if total > 0:
        print(f"\n  Crossover accuracy: {matches}/{total} = {matches / total:.1%}")
    print()

    # 4. Per-benchmark accuracy summary
    print("## Per-Benchmark Results\n")
    for benchmark in sorted(results.keys()):
        agg = results[benchmark].get("aggregate", {})
        print(f"### {benchmark}")
        strats = [(s, info.get("mean_accuracy", 0), info.get("std_accuracy", 0))
                  for s, info in agg.items()
                  if isinstance(info, dict) and "mean_accuracy" in info]
        strats.sort(key=lambda x: x[1], reverse=True)
        for s, acc, std in strats:
            print(f"  {s:35s}  {acc * 100:5.1f}% ± {std * 100:4.1f}%")
        print()

    # 5. Save full analysis
    report = {
        "results_dir": results_dir,
        "topology": topology,
        "crossover": crossover,
        "per_benchmark": {},
    }
    for b in results:
        agg = results[b].get("aggregate", {})
        report["per_benchmark"][b] = {
            s: {"mean_accuracy": info.get("mean_accuracy", 0), "std_accuracy": info.get("std_accuracy", 0)}
            for s, info in agg.items()
            if isinstance(info, dict) and "mean_accuracy" in info
        }

    out = output_path or os.path.join(results_dir, "analysis_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)
    generate_report(sys.argv[1])
