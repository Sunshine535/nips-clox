#!/usr/bin/env python3
"""Strategy Disagreement Analysis: can simple features predict which strategy is correct?

For each (model, benchmark) cell with Oracle−SC gap ≥ 0.15:
  1. Identify disagreement instances (strategies give different answers)
  2. Extract features per instance
  3. Train a per-disagreement selector (logistic regression)
  4. Measure: what fraction of the Oracle−SC gap does the selector capture?

Oracle gap capture =
    (SelectorAccuracy − SC) / (Oracle − SC)

Paths:
  >= 0.50 → strong positive (method paper)
  <= 0.10 → strong negative (unreachable-oracle paper)
  else → mixed (iterate)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print("Warning: sklearn not installed")

STRATEGIES = ["standard_cot", "self_consistency", "backward_cloze"]


def load_meta(base="results/meta"):
    cells = {}
    for d in sorted(Path(base).iterdir()):
        if not d.is_dir():
            continue
        f = d / "sweep_results.json"
        if f.exists():
            data = json.load(open(f))
            for bench, cell in data.get("cells", {}).items():
                key = (d.name, bench)
                cells[key] = cell
    return cells


def normalize_answer(a):
    if not a:
        return ""
    a = str(a).strip().lower()
    a = re.sub(r"[,\$%]", "", a)
    a = re.sub(r"\s+", " ", a).strip()
    return a


def extract_features(rows_by_strategy, i):
    """Feature vector for disagreement instance i."""
    cot = rows_by_strategy["standard_cot"][i]
    sc = rows_by_strategy["self_consistency"][i]
    bc = rows_by_strategy["backward_cloze"][i]

    def get(row, key, default=0.0):
        v = row.get(key, default)
        if v is None:
            return default
        return float(v)

    feats = {
        # Token counts
        "cot_tokens": get(cot, "total_tokens"),
        "sc_tokens": get(sc, "total_tokens"),
        "bc_tokens": get(bc, "total_tokens"),
        # Prediction length (characters)
        "cot_pred_len": len(str(cot.get("prediction", ""))),
        "sc_pred_len": len(str(sc.get("prediction", ""))),
        "bc_pred_len": len(str(bc.get("prediction", ""))),
        # Pairwise agreement
        "cot_eq_sc": int(normalize_answer(cot.get("prediction")) == normalize_answer(sc.get("prediction"))),
        "cot_eq_bc": int(normalize_answer(cot.get("prediction")) == normalize_answer(bc.get("prediction"))),
        "sc_eq_bc": int(normalize_answer(sc.get("prediction")) == normalize_answer(bc.get("prediction"))),
        # How many strategies agree with each
        # (equivalent to confidence)
    }

    # Pred-has features
    for sname, row in [("cot", cot), ("sc", sc), ("bc", bc)]:
        p = str(row.get("prediction", "")).lower()
        feats[f"{sname}_has_num"] = int(bool(re.search(r"\d", p)))
        feats[f"{sname}_is_yn"] = int(p.strip() in ("yes", "no", "true", "false"))
        feats[f"{sname}_has_letter_choice"] = int(bool(re.search(r"^\(?[abcde]\)?$", p.strip())))

    return feats


def label(rows_by_strategy, i):
    """Which strategy gets it right? Returns dict with per-strategy correct flag."""
    return {
        s: int(bool(rows_by_strategy[s][i].get("correct", False)))
        for s in STRATEGIES
    }


def main():
    cells = load_meta("results/meta")
    print(f"Loaded {len(cells)} cells")

    # Select good cells: Oracle - SC >= 0.15
    good_cells = [
        (k, c) for k, c in cells.items()
        if c.get("oracle_sc_gap", 0) >= 0.15
    ]
    print(f"Good cells (gap >= 0.15): {len(good_cells)}")
    for (m, b), c in good_cells:
        print(f"  {m}/{b}: gap={c['oracle_sc_gap']:.3f}")

    # For each good cell, build disagreement dataset
    all_rows = []
    for (m, b), cell in good_cells:
        rows_by_strategy = {s: cell["per_strategy_rows"][s] for s in STRATEGIES}
        n = cell["n_examples"]
        for i in range(n):
            labels = label(rows_by_strategy, i)
            # Include only DISAGREEMENT instances (not all 3 agree on correctness)
            # Actually we want instances where predictions differ, so we have a selection task
            preds = [
                normalize_answer(rows_by_strategy[s][i].get("prediction", ""))
                for s in STRATEGIES
            ]
            # How many unique predictions (1 = full agreement, 2 or 3 = disagreement)
            n_uniq = len(set(p for p in preds if p))
            feats = extract_features(rows_by_strategy, i)
            feats.update({
                "cell_model": m,
                "cell_benchmark": b,
                "example_id": rows_by_strategy["standard_cot"][i].get("example_id", ""),
                "cot_correct": labels["standard_cot"],
                "sc_correct": labels["self_consistency"],
                "bc_correct": labels["backward_cloze"],
                "n_unique_preds": n_uniq,
                "any_correct": int(labels["standard_cot"] or labels["self_consistency"] or labels["backward_cloze"]),
            })
            # Target: which strategy is correct when SC is wrong
            # (we want selector to improve over SC baseline)
            feats["sc_wrong_but_others_right"] = int(
                labels["self_consistency"] == 0 and
                (labels["standard_cot"] == 1 or labels["backward_cloze"] == 1)
            )
            # The "winner" in disagreement (if multiple strategies are correct, pick first)
            winners = [s for s in STRATEGIES if labels[s]]
            feats["winner"] = winners[0] if winners else "none"
            all_rows.append(feats)

    print(f"\nTotal instances: {len(all_rows)}")
    print(f"Disagreement instances (n_unique_preds >= 2): "
          f"{sum(1 for r in all_rows if r['n_unique_preds'] >= 2)}")
    print(f"SC-wrong-but-others-right: "
          f"{sum(1 for r in all_rows if r['sc_wrong_but_others_right'])}")

    # Save
    Path("results/disagreement").mkdir(parents=True, exist_ok=True)
    with open("results/disagreement/instances.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print("Saved: results/disagreement/instances.json")

    # === Part 2: Selector training via leave-one-cell-out ===
    if not HAS_SK:
        print("sklearn missing; skip training")
        return

    # For each good cell, build train/test split
    cell_groups = defaultdict(list)
    for r in all_rows:
        cell_groups[(r["cell_model"], r["cell_benchmark"])].append(r)

    feature_names = [
        "cot_tokens", "sc_tokens", "bc_tokens",
        "cot_pred_len", "sc_pred_len", "bc_pred_len",
        "cot_eq_sc", "cot_eq_bc", "sc_eq_bc",
        "cot_has_num", "sc_has_num", "bc_has_num",
        "cot_is_yn", "sc_is_yn", "bc_is_yn",
        "cot_has_letter_choice", "sc_has_letter_choice", "bc_has_letter_choice",
    ]

    print(f"\n{'=' * 60}")
    print("Leave-One-Cell-Out evaluation")
    print(f"{'=' * 60}")
    print(f"{'Test Cell':<30} {'SC':>6} {'Oracle':>7} {'Sel-LR':>7} {'Sel-GB':>7} {'Cap-LR':>7} {'Cap-GB':>7}")

    cell_list = list(cell_groups.keys())
    summary = []

    for test_cell in cell_list:
        train_rows = [r for ck, rs in cell_groups.items() if ck != test_cell for r in rs]
        test_rows = cell_groups[test_cell]

        X_train = np.array([[r[f] for f in feature_names] for r in train_rows])
        # Target: is SC correct? (We predict whether to trust SC or switch)
        # Actually: pick the strategy with highest predicted score.
        # Simpler framing: train 3 separate classifiers (one per strategy) to
        # predict per-strategy correctness; at inference, pick argmax.
        predictions = {}
        for sname in STRATEGIES:
            key = {
                "standard_cot": "cot_correct",
                "self_consistency": "sc_correct",
                "backward_cloze": "bc_correct",
            }[sname]
            y = np.array([r[key] for r in train_rows])
            if y.sum() == 0 or y.sum() == len(y):
                predictions[sname] = ("const", y[0] if len(y) else 0)
                continue
            clf = LogisticRegression(max_iter=1000).fit(X_train, y)
            predictions[sname] = ("lr", clf)

        X_test = np.array([[r[f] for f in feature_names] for r in test_rows])
        # Predict probability of correctness for each strategy
        probs = {}
        for sname, (kind, model) in predictions.items():
            if kind == "const":
                probs[sname] = np.full(len(X_test), model)
            else:
                probs[sname] = model.predict_proba(X_test)[:, 1]

        # Pick argmax per instance
        stack = np.stack([probs[s] for s in STRATEGIES], axis=1)
        picks = np.argmax(stack, axis=1)
        selector_correct = 0
        for j, r in enumerate(test_rows):
            pick_s = STRATEGIES[picks[j]]
            key = {
                "standard_cot": "cot_correct",
                "self_consistency": "sc_correct",
                "backward_cloze": "bc_correct",
            }[pick_s]
            selector_correct += r[key]
        sel_lr = selector_correct / len(test_rows)

        # Also try GB
        gb_predictions = {}
        for sname in STRATEGIES:
            key = {
                "standard_cot": "cot_correct",
                "self_consistency": "sc_correct",
                "backward_cloze": "bc_correct",
            }[sname]
            y = np.array([r[key] for r in train_rows])
            if y.sum() == 0 or y.sum() == len(y):
                gb_predictions[sname] = ("const", y[0] if len(y) else 0)
                continue
            clf = GradientBoostingClassifier(n_estimators=50, max_depth=3).fit(X_train, y)
            gb_predictions[sname] = ("gb", clf)

        probs_gb = {}
        for sname, (kind, model) in gb_predictions.items():
            if kind == "const":
                probs_gb[sname] = np.full(len(X_test), model)
            else:
                probs_gb[sname] = model.predict_proba(X_test)[:, 1]
        stack_gb = np.stack([probs_gb[s] for s in STRATEGIES], axis=1)
        picks_gb = np.argmax(stack_gb, axis=1)
        gb_correct = 0
        for j, r in enumerate(test_rows):
            pick_s = STRATEGIES[picks_gb[j]]
            key = {
                "standard_cot": "cot_correct",
                "self_consistency": "sc_correct",
                "backward_cloze": "bc_correct",
            }[pick_s]
            gb_correct += r[key]
        sel_gb = gb_correct / len(test_rows)

        # Baselines from the cell
        cell = cells[test_cell]
        sc_acc = cell["sc_accuracy"]
        oracle_acc = cell["oracle_accuracy"]
        gap = oracle_acc - sc_acc

        cap_lr = (sel_lr - sc_acc) / gap if gap > 0 else 0
        cap_gb = (sel_gb - sc_acc) / gap if gap > 0 else 0

        label_txt = f"{test_cell[0]}/{test_cell[1]}"
        print(f"{label_txt:<30} {sc_acc:>6.3f} {oracle_acc:>7.3f} {sel_lr:>7.3f} {sel_gb:>7.3f} "
              f"{cap_lr:>+7.2f} {cap_gb:>+7.2f}")
        summary.append({
            "test_cell": f"{test_cell[0]}/{test_cell[1]}",
            "sc_acc": sc_acc,
            "oracle_acc": oracle_acc,
            "sel_lr_acc": sel_lr,
            "sel_gb_acc": sel_gb,
            "gap": gap,
            "cap_lr": cap_lr,
            "cap_gb": cap_gb,
        })

    # Aggregate
    avg_cap_lr = np.mean([s["cap_lr"] for s in summary])
    avg_cap_gb = np.mean([s["cap_gb"] for s in summary])
    print(f"\n{'Average':<30} {'':<6} {'':<7} {'':<7} {'':<7} {avg_cap_lr:>+7.2f} {avg_cap_gb:>+7.2f}")

    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    max_cap = max(avg_cap_lr, avg_cap_gb)
    if max_cap >= 0.5:
        print(f">>> STRONG POSITIVE: selector captures {max_cap:.1%} of oracle gap on avg.")
        print("    Proceed with method paper.")
    elif max_cap <= 0.1:
        print(f">>> STRONG NEGATIVE: selector captures only {max_cap:.1%}.")
        print("    Oracle gap is unreachable with simple features → negative/diagnostic paper.")
    else:
        print(f">>> MARGINAL: selector captures {max_cap:.1%}. Needs more features or bigger data.")

    # Save
    with open("results/disagreement/selector_results.json", "w") as f:
        json.dump({
            "summary": summary,
            "avg_cap_lr": avg_cap_lr,
            "avg_cap_gb": avg_cap_gb,
            "verdict": ("positive" if max_cap >= 0.5 else
                       ("negative" if max_cap <= 0.1 else "marginal")),
        }, f, indent=2, default=str)
    print(f"\nSaved: results/disagreement/selector_results.json")


if __name__ == "__main__":
    main()
