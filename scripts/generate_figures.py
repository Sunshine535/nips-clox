#!/usr/bin/env python3
"""Generate publication-quality figures for CLOX paper."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

STRATEGY_DISPLAY = {
    "standard_cot": "Standard CoT",
    "self_consistency": "Self-Consistency",
    "backward_cloze": "Backward Cloze",
    "uncertainty_masked_repair": "Uncertainty Repair",
    "random_masked_repair": "Random Repair",
    "full_regeneration": "Full Regeneration",
    "hierarchical_repair": "Hierarchical Repair",
    "clox_adaptive": "CLOX-Adaptive",
}

STRATEGY_COLORS = {
    "standard_cot": "#4C72B0",
    "self_consistency": "#DD8452",
    "backward_cloze": "#55A868",
    "uncertainty_masked_repair": "#C44E52",
    "random_masked_repair": "#8172B3",
    "full_regeneration": "#937860",
    "hierarchical_repair": "#DA8BC3",
    "clox_adaptive": "#000000",
}


def load_results(results_dir: str) -> dict:
    results = {}
    for f in Path(results_dir).glob("*_results.json"):
        benchmark = f.stem.replace("_results", "")
        with open(f) as fh:
            results[benchmark] = json.load(fh)
    return results


def fig1_crossover_bar(results: dict, output_dir: str):
    """Figure 1: Bar chart showing strategy performance across benchmarks (crossover effect)."""
    benchmarks = sorted(results.keys())
    strategies = results[benchmarks[0]].get("strategy_names", [])

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(4 * len(benchmarks), 4), sharey=True)
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, benchmark in zip(axes, benchmarks):
        accs = []
        stds = []
        colors = []
        labels = []
        for s in strategies:
            agg = results[benchmark].get("aggregate", {}).get(s, {})
            accs.append(agg.get("mean_accuracy", 0) * 100)
            stds.append(agg.get("std_accuracy", 0) * 100)
            colors.append(STRATEGY_COLORS.get(s, "#666"))
            labels.append(STRATEGY_DISPLAY.get(s, s))

        x = np.arange(len(strategies))
        bars = ax.bar(x, accs, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(benchmark.replace("_", " ").upper(), fontweight="bold")
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

        # Highlight best
        best_idx = np.argmax(accs)
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_crossover_bar.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


def fig2_topology_scatter(results: dict, output_dir: str):
    """Figure 2: Scatter plot of (EPL, r̄) colored by best strategy, validating Theorem 1-2."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for benchmark, data in results.items():
        per_strat = data.get("per_strategy_results", {})
        adaptive_results = per_strat.get("clox_adaptive", {})
        for seed_key, examples in adaptive_results.items():
            for ex in examples:
                topo = ex.get("topology", {})
                epl = topo.get("epl")
                rbar = topo.get("recoverability")
                if epl is None or rbar is None:
                    continue
                ax.scatter(epl, rbar, alpha=0.3, s=15,
                          label=benchmark if benchmark not in [h.get_text() for h in ax.get_legend_handles_labels()[1]] else "",
                          color=STRATEGY_COLORS.get(topo.get("recommendation", ""), "#666"))

    # Theory boundaries
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="r̄ = 0.5 boundary")
    ax.axvline(x=3.0, color="gray", linestyle=":", alpha=0.5, label="EPL = 3.0 boundary")

    # Theory regions
    ax.text(1.5, 0.8, "Masking\nAdvantage\n(Theorem 1)", ha="center", fontsize=8, color="#55A868", fontweight="bold")
    ax.text(5.0, 0.2, "SC\nDominance\n(Theorem 2)", ha="center", fontsize=8, color="#DD8452", fontweight="bold")

    ax.set_xlabel("Error Propagation Length (EPL)")
    ax.set_ylabel("Local Recoverability (r̄)")
    ax.set_title("Task Topology and Strategy Selection")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.2)

    path = os.path.join(output_dir, "fig2_topology_scatter.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


def fig3_adaptive_gap(results: dict, output_dir: str):
    """Figure 3: CLOX-Adaptive accuracy vs oracle (best fixed strategy per benchmark)."""
    benchmarks = sorted(results.keys())
    strategies = results[benchmarks[0]].get("strategy_names", [])

    adaptive_accs = []
    oracle_accs = []
    cot_accs = []
    sc_accs = []
    labels = []

    for b in benchmarks:
        agg = results[b].get("aggregate", {})
        best_fixed = max(
            (agg.get(s, {}).get("mean_accuracy", 0) for s in strategies if s != "clox_adaptive"),
            default=0,
        )
        adaptive = agg.get("clox_adaptive", {}).get("mean_accuracy", 0)
        cot = agg.get("standard_cot", {}).get("mean_accuracy", 0)
        sc = agg.get("self_consistency", {}).get("mean_accuracy", 0)

        oracle_accs.append(best_fixed * 100)
        adaptive_accs.append(adaptive * 100)
        cot_accs.append(cot * 100)
        sc_accs.append(sc * 100)
        labels.append(b.replace("_", " ").upper())

    x = np.arange(len(benchmarks))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(6, len(benchmarks) * 1.5), 4))

    ax.bar(x - 1.5 * width, cot_accs, width, label="Standard CoT", color=STRATEGY_COLORS["standard_cot"])
    ax.bar(x - 0.5 * width, sc_accs, width, label="Self-Consistency", color=STRATEGY_COLORS["self_consistency"])
    ax.bar(x + 0.5 * width, adaptive_accs, width, label="CLOX-Adaptive", color=STRATEGY_COLORS["clox_adaptive"])
    ax.bar(x + 1.5 * width, oracle_accs, width, label="Oracle (Best Fixed)", color="#AAA", edgecolor="red", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CLOX-Adaptive vs. Fixed Strategies")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(output_dir, "fig3_adaptive_gap.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


def generate_all(results_dir: str, output_dir: str | None = None):
    results = load_results(results_dir)
    if not results:
        print(f"No results in {results_dir}")
        return

    out = output_dir or os.path.join(results_dir, "..", "..", "paper", "figures")
    os.makedirs(out, exist_ok=True)

    print(f"Generating figures for {len(results)} benchmarks → {out}")
    fig1_crossover_bar(results, out)
    fig2_topology_scatter(results, out)
    fig3_adaptive_gap(results, out)
    print("All figures generated.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir> [output_dir]")
        sys.exit(1)
    generate_all(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
