#!/usr/bin/env python3
"""Analyze CLOX v2 experiment results and generate paper figures."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed, skipping figures")


def load_results(results_dir: str) -> dict[str, Any]:
    """Load all result files from a model's output directory."""
    data = {}
    for f in Path(results_dir).rglob("*.json"):
        if f.name.startswith(".ckpt"):
            continue
        key = f.stem
        with open(f) as fh:
            data[key] = json.load(fh)
    return data


def summarize_topology(data: dict) -> str:
    """Generate topology summary table."""
    lines = []
    for key, content in sorted(data.items()):
        if not key.endswith("_topology"):
            continue
        s = content.get("summary", {})
        bench = s.get("benchmark", key)
        lines.append(
            f"| {bench:15s} | {s.get('r_bar_mean', 0):.3f} ± {s.get('r_bar_std', 0):.3f} "
            f"| {s.get('epl_mean', 0):.2f} ± {s.get('epl_std', 0):.2f} "
            f"| {s.get('n_examples', 0):5d} |"
        )
    header = f"| {'Benchmark':15s} | {'r̄ (mean ± std)':20s} | {'ℓ (mean ± std)':18s} | {'N':>5s} |"
    sep = "|" + "-" * 17 + "|" + "-" * 22 + "|" + "-" * 20 + "|" + "-" * 7 + "|"
    return "\n".join([header, sep] + lines)


def summarize_strategies(data: dict) -> str:
    """Generate strategy comparison table."""
    lines = []
    for key, content in sorted(data.items()):
        if not key.endswith("_strategies"):
            continue
        agg = content.get("aggregate", {})
        bench = content.get("benchmark", key)
        header_printed = False

        for sname in content.get("strategies", []):
            metrics = agg.get(sname, {})
            acc = metrics.get("mean_accuracy", 0)
            std = metrics.get("std_accuracy", 0)
            tokens = metrics.get("mean_tokens", 0)
            eff = metrics.get("token_efficiency", {})
            tpc = eff.get("tokens_per_correct", 0)

            prefix = f"  {bench}" if not header_printed else "  "
            header_printed = True
            lines.append(
                f"| {prefix:15s} | {sname:25s} | {acc:.4f} ± {std:.4f} "
                f"| {tokens:8.0f} | {tpc:10.0f} |"
            )
        lines.append("|" + "-" * 100 + "|")

    return "\n".join(lines)


def plot_phase_diagram(data: dict, output_path: str):
    """Plot the (r̄, ℓ) phase diagram with strategy regions."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Per-example scatter colored by recommended strategy
    ax = axes[0]
    colors = {
        "targeted_repair": "#2196F3",
        "self_consistency": "#F44336",
        "hierarchical_repair": "#FF9800",
        "standard_cot": "#4CAF50",
        "adaptive": "#9C27B0",
    }
    markers = {
        "gsm8k": "o", "math": "s", "strategyqa": "^", "arc_challenge": "D",
    }

    for key, content in sorted(data.items()):
        if not key.endswith("_topology"):
            continue
        bench = content.get("summary", {}).get("benchmark", "")
        for ex in content.get("per_example", []):
            r = ex.get("r_bar", 0.5)
            l = ex.get("epl", 3.0)
            s = ex.get("recommended_strategy", "adaptive")
            c = colors.get(s, "#999999")
            m = markers.get(bench, "o")
            ax.scatter(r, l, c=c, marker=m, s=30, alpha=0.5, edgecolors="none")

    # Add strategy region boundaries
    ax.axvline(x=0.65, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0.45, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")

    ax.set_xlabel("Local Recoverability (r̄)", fontsize=12)
    ax.set_ylabel("Normalized EPL (ℓ/n)", fontsize=12)
    ax.set_title("Phase Diagram: Strategy Selection by Topology", fontsize=13)

    patches = [mpatches.Patch(color=c, label=s) for s, c in colors.items()]
    bench_markers = [plt.Line2D([0], [0], marker=m, color='gray', linestyle='',
                                label=b, markersize=8)
                     for b, m in markers.items()]
    ax.legend(handles=patches + bench_markers, fontsize=8, loc="upper right")

    # Right: Per-benchmark r̄ vs ℓ mean ± std
    ax = axes[1]
    for key, content in sorted(data.items()):
        if not key.endswith("_topology"):
            continue
        s = content.get("summary", {})
        bench = s.get("benchmark", "")
        ax.errorbar(
            s.get("r_bar_mean", 0.5), s.get("epl_mean", 3.0),
            xerr=s.get("r_bar_std", 0), yerr=s.get("epl_std", 0),
            fmt=markers.get(bench, "o"), markersize=12, capsize=5,
            label=bench, linewidth=2,
        )

    ax.set_xlabel("Local Recoverability (r̄)", fontsize=12)
    ax.set_ylabel("Error Propagation Length (ℓ)", fontsize=12)
    ax.set_title("Benchmark Topology Profiles", fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved phase diagram to {output_path}")


def plot_strategy_comparison(data: dict, output_path: str):
    """Bar chart comparing strategies across benchmarks."""
    if not HAS_MPL:
        return

    # Collect data
    benchmarks = []
    strategy_names = []
    results = {}

    for key, content in sorted(data.items()):
        if not key.endswith("_strategies"):
            continue
        bench = content.get("benchmark", key)
        benchmarks.append(bench)
        agg = content.get("aggregate", {})
        for sname in content.get("strategies", []):
            if sname not in strategy_names:
                strategy_names.append(sname)
            metrics = agg.get(sname, {})
            results[(bench, sname)] = (
                metrics.get("mean_accuracy", 0),
                metrics.get("std_accuracy", 0),
            )

    if not benchmarks or not strategy_names:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(benchmarks) * 3), 7))
    x = np.arange(len(benchmarks))
    width = 0.8 / max(len(strategy_names), 1)

    colors_list = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))

    for i, sname in enumerate(strategy_names):
        accs = [results.get((b, sname), (0, 0))[0] for b in benchmarks]
        stds = [results.get((b, sname), (0, 0))[1] for b in benchmarks]
        offset = (i - len(strategy_names) / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, yerr=stds, label=sname,
               color=colors_list[i], capsize=2, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Strategy Comparison Across Benchmarks", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved strategy comparison to {output_path}")


def plot_compute_efficiency(data: dict, output_path: str):
    """Accuracy vs compute (tokens) scatter."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for key, content in sorted(data.items()):
        if not key.endswith("_strategies"):
            continue
        bench = content.get("benchmark", key)
        agg = content.get("aggregate", {})

        for i, sname in enumerate(content.get("strategies", [])):
            metrics = agg.get(sname, {})
            acc = metrics.get("mean_accuracy", 0)
            tokens = metrics.get("mean_tokens", 0)
            if tokens > 0:
                ax.scatter(tokens, acc, s=100, c=[colors[i % 10]],
                           label=f"{sname}" if bench == content.get("benchmark") else "",
                           alpha=0.7, edgecolors="black", linewidth=0.5)
                ax.annotate(f"{bench[:3]}", (tokens, acc), fontsize=6,
                           textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Mean Tokens per Example", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Compute Cost", fontsize=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved compute efficiency to {output_path}")


def generate_latex_table(data: dict) -> str:
    """Generate LaTeX table for the paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Accuracy comparison across benchmarks and strategies. Best result per benchmark in \textbf{bold}.}",
        r"\label{tab:main_results}",
        r"\small",
    ]

    benchmarks = []
    strategies = []
    results = {}

    for key, content in sorted(data.items()):
        if not key.endswith("_strategies"):
            continue
        bench = content.get("benchmark", key)
        benchmarks.append(bench)
        agg = content.get("aggregate", {})
        for sname in content.get("strategies", []):
            if sname not in strategies:
                strategies.append(sname)
            m = agg.get(sname, {})
            results[(bench, sname)] = (
                m.get("mean_accuracy", 0),
                m.get("std_accuracy", 0),
            )

    ncols = len(benchmarks) + 1
    lines.append(r"\begin{tabular}{l" + "c" * len(benchmarks) + "}")
    lines.append(r"\toprule")
    header = "Strategy & " + " & ".join(benchmarks) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for sname in strategies:
        best_per_bench = {}
        for b in benchmarks:
            best = max(results.get((b, s), (0, 0))[0] for s in strategies)
            best_per_bench[b] = best

        cells = [sname.replace("_", r"\_")]
        for b in benchmarks:
            acc, std = results.get((b, sname), (0, 0))
            cell = f"{acc:.1%}"
            if abs(acc - best_per_bench[b]) < 0.001:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_v2.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    data = load_results(results_dir)
    print(f"Loaded {len(data)} result files from {results_dir}")

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Text summaries
    print("\n=== TOPOLOGY SUMMARY ===")
    print(summarize_topology(data))

    print("\n=== STRATEGY COMPARISON ===")
    print(summarize_strategies(data))

    # LaTeX
    latex = generate_latex_table(data)
    latex_path = os.path.join(results_dir, "main_results.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {latex_path}")

    # Figures
    if HAS_MPL:
        plot_phase_diagram(data, os.path.join(fig_dir, "phase_diagram.pdf"))
        plot_strategy_comparison(data, os.path.join(fig_dir, "strategy_comparison.pdf"))
        plot_compute_efficiency(data, os.path.join(fig_dir, "compute_efficiency.pdf"))

    # Save full analysis
    analysis = {
        "topology": {},
        "strategies": {},
    }
    for key, content in data.items():
        if key.endswith("_topology"):
            analysis["topology"][key] = content.get("summary", {})
        if key.endswith("_strategies"):
            analysis["strategies"][key] = content.get("aggregate", {})

    analysis_path = os.path.join(results_dir, "analysis_summary.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
