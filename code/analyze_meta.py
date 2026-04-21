#!/usr/bin/env python3
"""Compute Oracle-SC gap across (model, benchmark) cells.

Usage: python3 code/analyze_meta.py results/meta
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_all(base: str) -> dict:
    out = {}
    for d in sorted(Path(base).iterdir()):
        if not d.is_dir():
            continue
        f = d / "sweep_results.json"
        if f.exists():
            with open(f) as fh:
                out[d.name] = json.load(fh)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_meta.py <meta_dir>"); sys.exit(1)
    base = sys.argv[1]
    data = load_all(base)
    if not data:
        print("No sweep_results.json found."); sys.exit(1)

    # Collect matrix
    models = list(data.keys())
    benchmarks = sorted({b for m in data.values() for b in m["cells"].keys()})

    print(f"Models: {models}")
    print(f"Benchmarks: {benchmarks}")
    print()

    # Build matrices
    def get(m, b, key):
        c = data[m]["cells"].get(b, {})
        return c.get(key, None)

    print("=" * 70)
    print("ORACLE accuracy")
    print("=" * 70)
    print(f"{'model':<18} " + " ".join(f"{b:>12}" for b in benchmarks))
    for m in models:
        row = []
        for b in benchmarks:
            v = get(m, b, "oracle_accuracy")
            row.append(f"{v:>12.3f}" if v is not None else f"{'-':>12}")
        print(f"{m:<18} " + " ".join(row))

    print("\n" + "=" * 70)
    print("SC(k=8) accuracy")
    print("=" * 70)
    print(f"{'model':<18} " + " ".join(f"{b:>12}" for b in benchmarks))
    for m in models:
        row = []
        for b in benchmarks:
            v = get(m, b, "sc_accuracy")
            row.append(f"{v:>12.3f}" if v is not None else f"{'-':>12}")
        print(f"{m:<18} " + " ".join(row))

    print("\n" + "=" * 70)
    print("ORACLE − SC gap (larger = more room for cross-strategy methods)")
    print("=" * 70)
    print(f"{'model':<18} " + " ".join(f"{b:>12}" for b in benchmarks))
    top_cells = []
    for m in models:
        row = []
        for b in benchmarks:
            v = get(m, b, "oracle_sc_gap")
            if v is not None:
                mark = "⭐" if v >= 0.15 else (" " if v >= 0.05 else "·")
                row.append(f"{v:>+11.3f}{mark}")
                top_cells.append((v, m, b))
            else:
                row.append(f"{'-':>12}")
        print(f"{m:<18} " + " ".join(row))

    print("\n" + "=" * 70)
    print("TOP CELLS BY GAP")
    print("=" * 70)
    top_cells.sort(reverse=True)
    for gap, m, b in top_cells[:8]:
        ora = get(m, b, "oracle_accuracy")
        sc = get(m, b, "sc_accuracy")
        per = get(m, b, "per_strategy_accuracy")
        print(f"  gap={gap:+.3f}  {m:<18} / {b:<15}  oracle={ora:.3f} sc={sc:.3f}  "
              f"cot={per.get('standard_cot', 0):.3f} bc={per.get('backward_cloze', 0):.3f}")

    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)
    max_gap = max(c[0] for c in top_cells) if top_cells else 0
    n_big = sum(1 for c in top_cells if c[0] >= 0.15)
    n_small = sum(1 for c in top_cells if c[0] < 0.05)
    print(f"Max gap: {max_gap:+.3f}")
    print(f"Cells with gap >= 0.15: {n_big}/{len(top_cells)}")
    print(f"Cells with gap < 0.05: {n_small}/{len(top_cells)}")
    if max_gap >= 0.15:
        print(f"\n>>> PROCEED: at least one cell has substantial room.")
        best = top_cells[0]
        print(f"    Focus next experiments on: {best[1]} / {best[2]} (gap={best[0]:+.3f})")
    elif n_small == len(top_cells):
        print("\n>>> PIVOT: all cells saturated, inference-time compute space is dead for this method family.")
    else:
        print(f"\n>>> INVESTIGATE: gap is marginal (max={max_gap:+.3f}). May need larger N to confirm.")

    # Heatmap
    if HAS_MPL:
        n_m, n_b = len(models), len(benchmarks)
        mat = np.full((n_m, n_b), np.nan)
        for i, m in enumerate(models):
            for j, b in enumerate(benchmarks):
                v = get(m, b, "oracle_sc_gap")
                if v is not None:
                    mat[i, j] = v
        fig, ax = plt.subplots(figsize=(max(7, n_b * 1.5), max(5, n_m * 0.6)))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.05, vmax=0.30, aspect="auto")
        ax.set_xticks(range(n_b))
        ax.set_yticks(range(n_m))
        ax.set_xticklabels(benchmarks, rotation=20, ha="right")
        ax.set_yticklabels(models)
        for i in range(n_m):
            for j in range(n_b):
                if not np.isnan(mat[i, j]):
                    v = mat[i, j]
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=9, color="black" if 0 <= v < 0.2 else "white")
        plt.colorbar(im, label="Oracle − SC gap")
        ax.set_title("Oracle − SC gap across model × benchmark")
        plt.tight_layout()
        out = os.path.join(base, "oracle_sc_gap_heatmap.pdf")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nHeatmap saved: {out}")

    # Save summary
    summary = {
        "max_gap": max_gap,
        "top_cells": [{"gap": g, "model": m, "benchmark": b} for g, m, b in top_cells[:8]],
        "cells_with_gap_ge_0.15": n_big,
        "cells_with_gap_lt_0.05": n_small,
        "decision": "PROCEED" if max_gap >= 0.15 else ("PIVOT" if n_small == len(top_cells) else "INVESTIGATE"),
    }
    with open(os.path.join(base, "meta_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
