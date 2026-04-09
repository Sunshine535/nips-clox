#!/usr/bin/env python3
"""Analyze cross-strategy pilot results for Idea A validation.

Analyses:
  1. Pairwise error correlation (phi coefficient) + heatmap
  2. PCA of error patterns (low-dimensional structure?)
  3. Cross-strategy voting vs SC (does diversity beat repetition?)
  4. Topology-error correlation (does r_bar predict disagreement?)
  5. Strategy agreement calibration (agreement ≈ accuracy?)
  6. Error transitivity (non-transitive triples = non-trivial structure)

Decision rule:
  |phi| < 0.3 between >= 3 pairs  -->  PROCEED
  |phi| > 0.7 for all pairs       -->  ABANDON
  Otherwise                        -->  INVESTIGATE

Usage:
    python analyze_pilot.py ../results/pilot
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations

import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import TwoSlopeNorm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Data Loading ───────────────────────────────────────────────────

def load_pilot(results_dir: str) -> dict:
    path = os.path.join(results_dir, "pilot_results.json")
    with open(path) as f:
        return json.load(f)


def build_error_matrix(data: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Build binary correctness matrix: (n_problems x n_strategies).

    Returns (matrix, problem_ids, strategy_names).
    """
    strategies = data["config"]["strategies"]
    results = data["strategy_results"]
    problems = [p["id"] for p in data["problems"]]

    matrix = np.zeros((len(problems), len(strategies)), dtype=int)
    for j, sname in enumerate(strategies):
        sres = results.get(sname, {})
        for i, pid in enumerate(problems):
            if pid in sres and sres[pid].get("correct"):
                matrix[i, j] = 1

    return matrix, problems, strategies


def get_token_costs(data: dict) -> dict[str, float]:
    """Mean tokens per example for each strategy."""
    costs = {}
    for sname, sres in data["strategy_results"].items():
        tokens = [v.get("total_tokens", 0) for v in sres.values()
                  if v.get("total_tokens", 0) > 0]
        costs[sname] = float(np.mean(tokens)) if tokens else 0.0
    return costs


# ── Analysis 1: Error Correlation ──────────────────────────────────

def phi_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Phi coefficient for two binary vectors."""
    a, b = x.astype(bool), y.astype(bool)
    n11 = int(np.sum(a & b))
    n00 = int(np.sum(~a & ~b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    denom = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    if denom < 1e-10:
        return 0.0
    return float(n11 * n00 - n10 * n01) / denom


def error_correlation_analysis(matrix: np.ndarray, strategies: list[str]) -> dict:
    """Compute pairwise phi coefficients and complementarity scores."""
    K = len(strategies)
    corr = np.zeros((K, K))
    complementarity = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i == j:
                corr[i, j] = 1.0
                complementarity[i, j] = 0.0
            else:
                corr[i, j] = phi_coefficient(matrix[:, i], matrix[:, j])
                # Fraction of problems where exactly one is correct
                complementarity[i, j] = float(
                    np.sum(matrix[:, i] != matrix[:, j])
                ) / matrix.shape[0]

    off_diag = corr[np.triu_indices_from(corr, k=1)]
    return {
        "phi_matrix": corr,
        "complementarity_matrix": complementarity,
        "n_low": int(np.sum(np.abs(off_diag) < 0.3)),
        "n_high": int(np.sum(np.abs(off_diag) > 0.7)),
        "n_pairs": len(off_diag),
        "mean_abs_phi": float(np.mean(np.abs(off_diag))),
        "median_abs_phi": float(np.median(np.abs(off_diag))),
        "min_phi": float(np.min(off_diag)),
        "max_phi": float(np.max(off_diag)),
    }


# ── Analysis 2: PCA ───────────────────────────────────────────────

def pca_analysis(matrix: np.ndarray, strategies: list[str]) -> dict:
    """PCA on error patterns — looking for low-dimensional structure."""
    centered = matrix.astype(float) - matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    total_var = np.sum(S ** 2)
    explained = (S ** 2) / total_var if total_var > 0 else S * 0
    cumulative = np.cumsum(explained)

    # 2D projection of problems
    proj = U[:, :2] * S[:2]

    # Strategy loadings on first 2 PCs
    loadings = Vt[:2].T  # shape (K, 2)

    n_80 = int(np.searchsorted(cumulative, 0.8) + 1)

    return {
        "explained_variance": explained.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "projection_2d": proj.tolist(),
        "loadings": loadings.tolist(),
        "n_components_80pct": n_80,
    }


# ── Analysis 3: Cross-Strategy Voting ─────────────────────────────

def cross_strategy_voting(
    matrix: np.ndarray, strategies: list[str], token_costs: dict[str, float],
) -> dict:
    """Compare cross-strategy majority voting vs SC."""
    # Single-answer strategies for voting (exclude SC variants)
    single = [s for s in strategies if s not in ("self_consistency", "compute_matched_sc")]
    single_idx = [strategies.index(s) for s in single]

    results = {}

    # Oracle: any strategy correct -> correct (ceiling)
    oracle_acc = float(np.any(matrix, axis=1).mean())
    results["oracle_any"] = {"acc": oracle_acc, "desc": "Any strategy correct"}

    # Full-vote: majority of all strategies
    full_vote = (matrix.sum(axis=1) > len(strategies) / 2).astype(int)
    results["full_vote"] = {"acc": float(full_vote.mean()),
                            "tokens": sum(token_costs.get(s, 0) for s in strategies)}

    # Cross-strategy voting at K=3,5,all_single
    for K in [3, 5, len(single)]:
        if K > len(single):
            continue

        combo_results = []
        for combo in combinations(range(len(single)), K):
            cols = [single_idx[c] for c in combo]
            votes = matrix[:, cols].sum(axis=1)
            majority = (votes > K / 2).astype(int)
            acc = float(majority.mean())
            cost = sum(token_costs.get(single[c], 0) for c in combo)
            combo_results.append({
                "acc": acc,
                "combo": [single[c] for c in combo],
                "tokens": cost,
            })

        combo_results.sort(key=lambda x: x["acc"], reverse=True)
        results[f"cross_K{K}"] = {
            "best": combo_results[0],
            "mean_acc": float(np.mean([c["acc"] for c in combo_results])),
            "worst": combo_results[-1],
            "n_combos": len(combo_results),
            "all_combos": combo_results,
        }

    # Baselines
    for sname in strategies:
        idx = strategies.index(sname)
        results[sname] = {
            "acc": float(matrix[:, idx].mean()),
            "tokens": token_costs.get(sname, 0),
        }

    return results


# ── Analysis 4: Topology-Error Correlation ─────────────────────────

def topology_error_analysis(
    matrix: np.ndarray, strategies: list[str], data: dict,
) -> dict:
    """Correlate topology (r_bar, epl) with strategy disagreement."""
    topology = data.get("topology", {})
    if not topology or not HAS_SCIPY:
        return {}

    problems = [p["id"] for p in data["problems"]]
    r_bars, epls, disagreements, diversities = [], [], [], []

    for i, pid in enumerate(problems):
        topo = topology.get(pid)
        if not topo:
            continue

        row = matrix[i]
        n_correct = row.sum()
        majority = 1 if n_correct > len(strategies) / 2 else 0
        disagree = int(np.sum(row != majority))
        diversity = float(min(n_correct, len(strategies) - n_correct)) / len(strategies)

        r_bars.append(topo["r_bar"])
        epls.append(topo["epl"])
        disagreements.append(disagree)
        diversities.append(diversity)

    if len(r_bars) < 5:
        return {}

    r_bars = np.array(r_bars)
    epls = np.array(epls)
    disagreements = np.array(disagreements)
    diversities = np.array(diversities)

    return {
        "r_bar_vs_disagreement": {
            "pearson_r": float(stats.pearsonr(r_bars, disagreements)[0]),
            "p_value": float(stats.pearsonr(r_bars, disagreements)[1]),
        },
        "epl_vs_disagreement": {
            "pearson_r": float(stats.pearsonr(epls, disagreements)[0]),
            "p_value": float(stats.pearsonr(epls, disagreements)[1]),
        },
        "r_bar_vs_diversity": {
            "pearson_r": float(stats.pearsonr(r_bars, diversities)[0]),
            "p_value": float(stats.pearsonr(r_bars, diversities)[1]),
        },
        "r_bars": r_bars.tolist(),
        "epls": epls.tolist(),
        "disagreements": disagreements.tolist(),
        "diversities": diversities.tolist(),
    }


# ── Analysis 5: Calibration ───────────────────────────────────────

def calibration_analysis(matrix: np.ndarray, strategies: list[str]) -> dict:
    """Strategy agreement as confidence calibration.

    For each problem, the agreement level = how many strategies agree
    with the majority answer. Higher agreement should mean higher accuracy.
    """
    K = len(strategies)
    buckets: dict[int, dict] = {}

    for i in range(matrix.shape[0]):
        n_correct = int(matrix[i].sum())
        n_agree = max(n_correct, K - n_correct)  # majority side count

        if n_agree not in buckets:
            buckets[n_agree] = {"n": 0, "majority_correct": 0}
        buckets[n_agree]["n"] += 1
        if n_correct > K / 2:
            buckets[n_agree]["majority_correct"] += 1

    calibration = {}
    for level, counts in sorted(buckets.items()):
        calibration[level] = {
            "n": counts["n"],
            "accuracy": counts["majority_correct"] / max(counts["n"], 1),
            "expected_confidence": level / K,
        }
    return calibration


# ── Analysis 6: Error Transitivity ─────────────────────────────────

def error_transitivity(corr: np.ndarray, strategies: list[str]) -> list[dict]:
    """Find non-transitive error triples: A~B, B~C, but A _|_ C.

    Non-transitivity = non-trivial structure in strategy space.
    """
    triples = []
    for a, b, c in combinations(range(len(strategies)), 3):
        rho_ab = corr[a, b]
        rho_bc = corr[b, c]
        rho_ac = corr[a, c]
        # A and B correlated, B and C correlated, but A and C independent
        if abs(rho_ab) > 0.25 and abs(rho_bc) > 0.25 and abs(rho_ac) < 0.15:
            triples.append({
                "A": strategies[a], "B": strategies[b], "C": strategies[c],
                "rho_AB": round(rho_ab, 3),
                "rho_BC": round(rho_bc, 3),
                "rho_AC": round(rho_ac, 3),
            })
    return triples


# ── Visualization ──────────────────────────────────────────────────

def plot_correlation_heatmap(
    corr: np.ndarray, strategies: list[str], output_path: str,
):
    if not HAS_MPL:
        return
    short = [s.replace("_", "\n") for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(0.5, np.max(np.abs(corr[np.triu_indices_from(corr, k=1)])) + 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(corr, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(strategies)))
    ax.set_yticks(range(len(strategies)))
    ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(short, fontsize=8)

    for i in range(len(strategies)):
        for j in range(len(strategies)):
            v = corr[i, j]
            color = "white" if abs(v) > 0.4 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, label="Phi Coefficient (error correlation)")
    ax.set_title("Pairwise Error Correlation Between Strategies", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pca(
    pca_res: dict, data: dict, matrix: np.ndarray, output_path: str,
):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ev = pca_res["explained_variance"]
    cumul = pca_res["cumulative_variance"]

    # Scree plot
    ax = axes[0]
    x = range(1, len(ev) + 1)
    ax.bar(x, ev, alpha=0.7, color="#2196F3", label="Individual")
    ax.plot(x, cumul, "ro-", label="Cumulative")
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA of Strategy Error Patterns")
    ax.legend()
    ax.set_xticks(list(x))

    # 2D scatter colored by difficulty
    ax = axes[1]
    proj = np.array(pca_res["projection_2d"])
    diff_colors = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#F44336"}

    for i, p in enumerate(data["problems"]):
        diff = p.get("difficulty", "medium")
        color = diff_colors.get(diff, "#999")
        n_correct = int(matrix[i].sum())
        ax.scatter(proj[i, 0], proj[i, 1], c=color,
                   s=20 + n_correct * 12, alpha=0.7,
                   edgecolors="black", linewidth=0.3)

    patches = [mpatches.Patch(color=c, label=d) for d, c in diff_colors.items()]
    ax.legend(handles=patches, title="Difficulty")
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)" if len(ev) > 1 else "PC2")
    ax.set_title("Problems in Strategy Error Space")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_voting_comparison(voting: dict, strategies: list[str], output_path: str):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 6))

    entries = []
    # CoT baseline
    if "standard_cot" in voting:
        entries.append(("CoT", voting["standard_cot"]["acc"], "#9E9E9E"))
    # SC baselines
    if "self_consistency" in voting:
        entries.append(("SC (K=8)", voting["self_consistency"]["acc"], "#F44336"))
    if "compute_matched_sc" in voting:
        entries.append(("SC (K=2)", voting["compute_matched_sc"]["acc"], "#E57373"))
    # Cross-strategy voting
    for key in sorted(voting.keys()):
        if key.startswith("cross_K"):
            K = key.split("K")[1]
            best = voting[key]["best"]
            entries.append((f"Cross-Vote\n(K={K}, best)", best["acc"], "#2196F3"))
            entries.append((f"Cross-Vote\n(K={K}, mean)", voting[key]["mean_acc"], "#90CAF9"))
    # Full vote
    if "full_vote" in voting:
        entries.append(("Full Vote\n(all 8)", voting["full_vote"]["acc"], "#1565C0"))
    # Oracle
    if "oracle_any" in voting:
        entries.append(("Oracle\n(any correct)", voting["oracle_any"]["acc"], "#4CAF50"))

    labels = [e[0] for e in entries]
    accs = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    bars = ax.bar(range(len(entries)), accs, color=colors,
                  edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Strategy Voting vs Self-Consistency")
    ax.set_ylim(0, min(1.05, max(accs) + 0.1))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_topology_diversity(topo: dict, output_path: str):
    if not HAS_MPL or not topo:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    r_bars = np.array(topo["r_bars"])
    epls = np.array(topo["epls"])
    disagree = np.array(topo["disagreements"])

    for ax, x, xlabel, key, color in [
        (axes[0], r_bars, "Recoverability (r-bar)", "r_bar_vs_disagreement", "#2196F3"),
        (axes[1], epls, "Error Propagation Length (epl)", "epl_vs_disagreement", "#FF9800"),
    ]:
        ax.scatter(x, disagree, alpha=0.6, c=color, edgecolors="black", linewidth=0.3)
        if len(x) >= 2 and np.std(x) > 1e-10:
            z = np.polyfit(x, disagree, 1)
            x_fit = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_fit, np.polyval(z, x_fit), "r--", alpha=0.7)
        info = topo.get(key, {})
        r_val = info.get("pearson_r", 0)
        p_val = info.get("p_value", 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Strategy Disagreement (# minority votes)")
        ax.set_title(f"r={r_val:.3f}, p={p_val:.3f}")

    fig.suptitle("Topology vs Strategy Diversity", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_pilot.py <results_dir>")
        print("  e.g. python analyze_pilot.py ../results/pilot")
        sys.exit(1)

    results_dir = sys.argv[1]
    data = load_pilot(results_dir)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    matrix, problems, strategies = build_error_matrix(data)
    token_costs = get_token_costs(data)
    N, K = matrix.shape
    print(f"Error matrix: {N} problems x {K} strategies")
    print(f"Overall accuracy: {matrix.mean():.1%}")

    # ── 1. Error Correlation ──
    print("\n" + "=" * 60)
    print("1. PAIRWISE ERROR CORRELATION (phi coefficient)")
    print("=" * 60)
    corr_analysis = error_correlation_analysis(matrix, strategies)
    phi = corr_analysis["phi_matrix"]

    for i in range(K):
        for j in range(i + 1, K):
            marker = ""
            if abs(phi[i, j]) < 0.3:
                marker = " <-- LOW"
            elif abs(phi[i, j]) > 0.7:
                marker = " <-- HIGH"
            print(f"  {strategies[i]:25s} vs {strategies[j]:25s}:  "
                  f"phi={phi[i, j]:+.3f}{marker}")

    print(f"\n  Low correlation (|phi| < 0.3): {corr_analysis['n_low']}/{corr_analysis['n_pairs']} pairs")
    print(f"  High correlation (|phi| > 0.7): {corr_analysis['n_high']}/{corr_analysis['n_pairs']} pairs")
    print(f"  Mean |phi|: {corr_analysis['mean_abs_phi']:.3f}")

    # ── 2. PCA ──
    print("\n" + "=" * 60)
    print("2. PCA OF ERROR PATTERNS")
    print("=" * 60)
    pca = pca_analysis(matrix, strategies)
    for i, (ev, c) in enumerate(zip(pca["explained_variance"], pca["cumulative_variance"])):
        print(f"  PC{i+1}: {ev:.1%} (cumulative: {c:.1%})")
    print(f"  Components for 80% variance: {pca['n_components_80pct']}")
    if pca["n_components_80pct"] <= 3:
        print("  --> LOW-DIMENSIONAL: error space has exploitable structure!")

    # Strategy loadings
    print("\n  Strategy loadings on PC1/PC2:")
    loadings = np.array(pca["loadings"])
    for j, sname in enumerate(strategies):
        print(f"    {sname:25s}  PC1={loadings[j, 0]:+.3f}  PC2={loadings[j, 1]:+.3f}")

    # ── 3. Cross-Strategy Voting ──
    print("\n" + "=" * 60)
    print("3. CROSS-STRATEGY VOTING vs SELF-CONSISTENCY")
    print("=" * 60)
    voting = cross_strategy_voting(matrix, strategies, token_costs)

    print(f"  Oracle (any correct):  {voting['oracle_any']['acc']:.1%}")
    print(f"  Full vote (all {K}):    {voting['full_vote']['acc']:.1%} "
          f" tokens={voting['full_vote']['tokens']:.0f}")
    print()

    for sname in strategies:
        if sname in voting:
            v = voting[sname]
            print(f"  {sname:25s}  acc={v['acc']:.1%}  tokens={v['tokens']:.0f}")
    print()

    for key in sorted(voting.keys()):
        if not key.startswith("cross_K"):
            continue
        v = voting[key]
        b = v["best"]
        print(f"  {key}: best={b['acc']:.1%} ({', '.join(b['combo'])}) "
              f"tokens={b['tokens']:.0f}")
        print(f"  {' ' * len(key)}  mean={v['mean_acc']:.1%}  "
              f"worst={v['worst']['acc']:.1%}  combos={v['n_combos']}")

    # Key comparison: best cross-K3 vs SC
    sc_acc = voting.get("self_consistency", {}).get("acc", 0)
    sc_tokens = voting.get("self_consistency", {}).get("tokens", 0)
    if "cross_K3" in voting:
        xv = voting["cross_K3"]["best"]
        delta = xv["acc"] - sc_acc
        token_ratio = xv["tokens"] / sc_tokens if sc_tokens > 0 else 0
        print(f"\n  KEY: Cross-Vote(K=3) vs SC(K=8): "
              f"delta={delta:+.1%}  token_ratio={token_ratio:.2f}x")

    # ── 4. Topology-Error Correlation ──
    print("\n" + "=" * 60)
    print("4. TOPOLOGY-ERROR CORRELATION")
    print("=" * 60)
    topo = topology_error_analysis(matrix, strategies, data)
    if topo:
        rd = topo["r_bar_vs_disagreement"]
        ed = topo["epl_vs_disagreement"]
        rv = topo["r_bar_vs_diversity"]
        print(f"  r_bar vs disagreement:  r={rd['pearson_r']:+.3f}  p={rd['p_value']:.3f}")
        print(f"  epl   vs disagreement:  r={ed['pearson_r']:+.3f}  p={ed['p_value']:.3f}")
        print(f"  r_bar vs diversity:     r={rv['pearson_r']:+.3f}  p={rv['p_value']:.3f}")
        if abs(rd["pearson_r"]) > 0.3 and rd["p_value"] < 0.05:
            print("  --> SIGNIFICANT: topology predicts strategy diversity!")
    else:
        print("  (no topology data or scipy not installed)")

    # ── 5. Calibration ──
    print("\n" + "=" * 60)
    print("5. STRATEGY AGREEMENT CALIBRATION")
    print("=" * 60)
    calib = calibration_analysis(matrix, strategies)
    print(f"  {'Agreement':>10s}  {'N':>4s}  {'Actual':>8s}  {'Expected':>8s}  {'Gap':>6s}")
    for level, c in sorted(calib.items()):
        gap = c["accuracy"] - c["expected_confidence"]
        print(f"  {level:>3d}/{K:<3d}      {c['n']:4d}  {c['accuracy']:7.1%}  "
              f"{c['expected_confidence']:8.1%}  {gap:+5.1%}")

    # ── 6. Error Transitivity ──
    print("\n" + "=" * 60)
    print("6. ERROR TRANSITIVITY")
    print("=" * 60)
    non_trans = error_transitivity(phi, strategies)
    print(f"  Non-transitive triples: {len(non_trans)}")
    for t in non_trans[:5]:
        print(f"    {t['A']:20s} ~ {t['B']:20s} (rho={t['rho_AB']:+.2f})")
        print(f"    {t['B']:20s} ~ {t['C']:20s} (rho={t['rho_BC']:+.2f})")
        print(f"    {t['A']:20s} | {t['C']:20s} (rho={t['rho_AC']:+.2f})")
        print()

    # ── Per-Strategy Summary ──
    print("=" * 60)
    print("PER-STRATEGY ACCURACY BY DIFFICULTY")
    print("=" * 60)
    for j, sname in enumerate(strategies):
        acc = matrix[:, j].mean()
        diff_accs: dict[str, list] = {}
        for i, p in enumerate(data["problems"]):
            d = p.get("difficulty", "medium")
            diff_accs.setdefault(d, []).append(matrix[i, j])
        parts = [f"{d}={np.mean(v):.0%}" for d, v in sorted(diff_accs.items())]
        print(f"  {sname:25s}  {acc:.1%}  [{', '.join(parts)}]  "
              f"tokens={token_costs.get(sname, 0):.0f}")

    # ── Decision ──
    print("\n" + "=" * 60)
    print("DECISION")
    print("=" * 60)
    n_low = corr_analysis["n_low"]
    n_high = corr_analysis["n_high"]
    n_pairs = corr_analysis["n_pairs"]
    mean_phi = corr_analysis["mean_abs_phi"]

    if n_low >= 3:
        decision = "PROCEED"
        reason = f"{n_low}/{n_pairs} pairs have low correlation (|phi|<0.3)"
    elif n_high == n_pairs:
        decision = "ABANDON"
        reason = "All pairs highly correlated (|phi|>0.7)"
    else:
        decision = "INVESTIGATE"
        reason = f"Moderate correlations: mean|phi|={mean_phi:.3f}"

    cross_wins = False
    if "cross_K3" in voting and sc_acc > 0:
        cross_wins = voting["cross_K3"]["best"]["acc"] > sc_acc
    if cross_wins:
        reason += "; cross-strategy voting BEATS SC"

    pca_low_dim = pca["n_components_80pct"] <= 3
    if pca_low_dim:
        reason += f"; error space is {pca['n_components_80pct']}-dimensional (low!)"

    print(f"\n  >>> {decision} <<<")
    print(f"  {reason}")
    print()

    # ── Figures ──
    if HAS_MPL:
        print("Generating figures...")
        plot_correlation_heatmap(phi, strategies, os.path.join(fig_dir, "error_correlation.pdf"))
        plot_pca(pca, data, matrix, os.path.join(fig_dir, "pca_error_space.pdf"))
        plot_voting_comparison(voting, strategies, os.path.join(fig_dir, "voting_comparison.pdf"))
        plot_topology_diversity(topo, os.path.join(fig_dir, "topology_diversity.pdf"))

    # ── Save Full Analysis ──
    analysis = {
        "summary": {
            "n_problems": N,
            "n_strategies": K,
            "overall_accuracy": float(matrix.mean()),
            "decision": decision,
            "reason": reason,
        },
        "error_correlation": {
            "phi_matrix": phi.tolist(),
            "n_low": n_low,
            "n_high": n_high,
            "n_pairs": n_pairs,
            "mean_abs_phi": mean_phi,
        },
        "pca": pca,
        "voting": {k: v for k, v in voting.items()
                   if not isinstance(v, dict) or "all_combos" not in v},
        "voting_detail": {k: v for k, v in voting.items()
                         if isinstance(v, dict) and "all_combos" in v},
        "topology_analysis": topo if topo else None,
        "calibration": {str(k): v for k, v in calib.items()},
        "non_transitive_triples": non_trans,
        "per_strategy": {
            sname: {
                "accuracy": float(matrix[:, j].mean()),
                "tokens": token_costs.get(sname, 0),
            }
            for j, sname in enumerate(strategies)
        },
    }

    analysis_path = os.path.join(results_dir, "pilot_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nFull analysis: {analysis_path}")


if __name__ == "__main__":
    main()
