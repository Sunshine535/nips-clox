#!/usr/bin/env python3
"""Question-Conditioned Strategy Router.

Core idea: BEFORE running any strategy, predict the best strategy from the
problem text alone. Route to that single strategy at inference time.

Compute advantage: at test time we run only 1 strategy (not SC's k=8).
Router overhead = zero forward passes.

Key difference vs Route-to-Reason: we use self-supervised labels from
cross-strategy agreement (no external training data).

Usage:
    python3 code/question_router.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

STRATEGIES = ["standard_cot", "self_consistency", "backward_cloze"]


def load_meta(base="results/meta"):
    """Load per-example rows from all meta-sweep cells with full question text."""
    examples = []
    for d in sorted(Path(base).iterdir()):
        if not d.is_dir():
            continue
        f = d / "sweep_results.json"
        if not f.exists():
            continue
        data = json.load(open(f))
        for bench, cell in data.get("cells", {}).items():
            rows = cell["per_strategy_rows"]
            n = cell["n_examples"]
            for i in range(n):
                # Pull question from any strategy's row (they all have the same id)
                ex_id = rows["standard_cot"][i].get("example_id", f"{bench}_{i}")
                gt = rows["standard_cot"][i].get("ground_truth", "")
                labels = {s: int(bool(rows[s][i].get("correct", False))) for s in STRATEGIES}
                # We don't have question text in the sweep output;
                # we need to load it from the problems list or original benchmark
                examples.append({
                    "model": d.name,
                    "benchmark": bench,
                    "example_id": ex_id,
                    "ground_truth": gt,
                    "correct": labels,
                    "n_correct": sum(labels.values()),
                    "cell_tokens": cell.get("per_strategy_mean_tokens", {}),
                })
    return examples


def load_benchmark_questions(benchmark_names):
    """Load question text for each example_id.
    We need to reconstruct problems from the original data source.
    """
    # Cache question text from our pilot/bav problems.json (same example_ids)
    questions = {}
    # Pilot problems (GSM8K)
    for p in [Path("results/pilot/problems.json"),
              Path("results/bav/problems.json")]:
        if p.exists():
            for item in json.load(open(p)):
                questions[item["id"]] = item["question"]
    # Also check shard problems
    for d in Path("results/pilot").glob("shard_*/problems.json"):
        for item in json.load(open(d)):
            questions[item["id"]] = item["question"]

    # For meta-sweep, we need to reload from HF datasets via same example_id scheme
    # example_id is formatted like "gsm8k_42", "strategyqa_17", etc.
    # We'll reload the raw benchmarks and map by example_id
    import sys
    sys.path.insert(0, "code")
    try:
        # Need to load from cache without triggering network
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = "/home/tarkoy/nips/nips-clox/results"  # won't work but OK
        # Actually try to use the offline cache
    except Exception:
        pass
    return questions


def text_features(q: str) -> dict:
    """Handcrafted text features from question alone."""
    q_lower = q.lower()
    q_len = len(q)
    words = q.split()
    n_words = len(words)

    # Digit/math features
    n_digits = sum(c.isdigit() for c in q)
    n_numbers = len(re.findall(r"\d+", q))
    has_math_op = int(bool(re.search(r"[\+\-\*\/=\^]", q)))
    has_dollar = int("$" in q)
    has_percent = int("%" in q)

    # Question-type markers
    is_yn = int(any(q.strip().startswith(w) for w in
                    ("is ", "are ", "can ", "do ", "does ", "did ", "will ",
                     "would ", "should ", "could ")) or
                any(m in q_lower for m in ("yes", "no,")))
    is_choice = int(bool(re.search(r"\([a-eA-E]\)", q)))

    # Reasoning-word features
    has_calc = int(any(w in q_lower for w in ("calculate", "compute", "total",
                                              "how many", "how much")))
    has_why = int("why" in q_lower)
    has_what = int("what" in q_lower)
    has_if = int("if " in q_lower or "if," in q_lower)
    has_logic = int(any(w in q_lower for w in ("logic", "statement",
                                               "follow", "premise", "deduce")))

    return {
        "q_len": q_len,
        "n_words": n_words,
        "n_digits": n_digits,
        "n_numbers": n_numbers,
        "has_math_op": has_math_op,
        "has_dollar": has_dollar,
        "has_percent": has_percent,
        "is_yn": is_yn,
        "is_choice": is_choice,
        "has_calc": has_calc,
        "has_why": has_why,
        "has_what": has_what,
        "has_if": has_if,
        "has_logic": has_logic,
    }


def assign_best_strategy(labels: dict) -> str:
    """Label: which strategy is "best" for this problem.

    Rules (ordered by preference when tied):
    1. If all three agree correct → SC (since it's standard)
    2. If only one correct → that one
    3. If two correct → prefer cheapest (CoT > BC > SC in token cost)
    """
    correct_strats = [s for s in STRATEGIES if labels[s]]
    if not correct_strats:
        return "none"  # no strategy worked
    if len(correct_strats) == 3:
        return "self_consistency"  # safe default when all agree
    if len(correct_strats) == 1:
        return correct_strats[0]
    # Two correct — prefer cheaper
    cost = {"standard_cot": 1, "backward_cloze": 4, "self_consistency": 8}
    return min(correct_strats, key=lambda s: cost[s])


def load_questions_from_datasets():
    """Load questions directly via datasets library with local cache."""
    import os
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/.cache/huggingface/datasets")
    # This is our local machine, no server cache. Skip — use fallback.
    return {}


def synthesize_questions(examples):
    """Fallback: reconstruct question text by loading benchmarks locally
    via HF datasets (cached locally if we have it)."""
    import sys
    sys.path.insert(0, "code")
    try:
        # Try to load from code/benchmarks.py
        from benchmarks import load_benchmark, load_math, load_bbh
    except Exception as e:
        print(f"Can't load benchmarks: {e}")
        return {}

    # Group by benchmark
    by_bench: dict[str, list] = defaultdict(list)
    for ex in examples:
        by_bench[ex["benchmark"]].append(ex["example_id"])

    q_map = {}
    for bname, ids in by_bench.items():
        print(f"Loading {bname}...")
        try:
            if bname == "math_hard":
                items = load_math(levels=[4, 5])
            elif bname == "bbh_logic":
                items = load_bbh(subtasks=["logical_deduction_five_objects"])
            else:
                items = load_benchmark(bname)
            for item in items:
                if item.example_id in ids:
                    q_map[item.example_id] = item.question
        except Exception as e:
            print(f"  Skip {bname}: {e}")
    return q_map


def main():
    examples = load_meta("results/meta")
    print(f"Loaded {len(examples)} instances from meta-sweep")

    # Restrict to good cells only (Oracle-SC gap >= 0.15)
    # Reconstruct gap from our data
    cells = {}
    for d in sorted(Path("results/meta").iterdir()):
        if not d.is_dir():
            continue
        f = d / "sweep_results.json"
        if f.exists():
            data = json.load(open(f))
            for bench, cell in data.get("cells", {}).items():
                cells[(d.name, bench)] = cell

    good = {k for k, c in cells.items() if c.get("oracle_sc_gap", 0) >= 0.15}
    examples = [e for e in examples if (e["model"], e["benchmark"]) in good]
    print(f"Good-cell instances: {len(examples)}")

    # Need questions
    q_map = synthesize_questions(examples)
    print(f"Resolved question text for {len(q_map)}/{len(examples)} examples")

    if len(q_map) < len(examples) * 0.5:
        # Fallback: use example_id as proxy — will be weaker but works
        print("Warning: using example_id substring as question text (degraded features)")
        for e in examples:
            if e["example_id"] not in q_map:
                q_map[e["example_id"]] = e["example_id"]

    # Build feature matrix
    data_rows = []
    for ex in examples:
        q = q_map.get(ex["example_id"], ex["example_id"])
        feats = text_features(q)
        feats["example_id"] = ex["example_id"]
        feats["benchmark"] = ex["benchmark"]
        feats["model"] = ex["model"]
        feats["best_strategy"] = assign_best_strategy(ex["correct"])
        feats["cot_correct"] = ex["correct"]["standard_cot"]
        feats["sc_correct"] = ex["correct"]["self_consistency"]
        feats["bc_correct"] = ex["correct"]["backward_cloze"]
        feats["any_correct"] = int(any(ex["correct"].values()))
        feats["_question"] = q
        data_rows.append(feats)

    # Distribution of best_strategy labels
    counts = Counter([r["best_strategy"] for r in data_rows])
    print(f"\nBest-strategy distribution: {dict(counts)}")

    feature_names = ["q_len", "n_words", "n_digits", "n_numbers",
                     "has_math_op", "has_dollar", "has_percent",
                     "is_yn", "is_choice", "has_calc", "has_why",
                     "has_what", "has_if", "has_logic"]

    # Strategy label encoder
    strat_labels = ["standard_cot", "self_consistency", "backward_cloze", "none"]
    label_to_idx = {l: i for i, l in enumerate(strat_labels)}

    # Leave-one-benchmark-out evaluation
    benches = sorted({r["benchmark"] for r in data_rows})
    print(f"\nBenchmarks for LOBO: {benches}")
    print(f"\n{'='*70}")
    print("LOBO Transfer Test: train on N-1 benchmarks, test on 1")
    print(f"{'='*70}")
    print(f"{'Test benchmark':<15} {'SC':>6} {'Oracle':>7} {'Rt-LR':>6} {'Rt-GB':>6} "
          f"{'Cap-LR':>7} {'Cap-GB':>7} {'Cost-LR':>8}")

    summary = []
    for test_b in benches:
        train = [r for r in data_rows if r["benchmark"] != test_b]
        test = [r for r in data_rows if r["benchmark"] == test_b]
        if not train or not test:
            continue

        X_tr_num = np.array([[r[f] for f in feature_names] for r in train], dtype=float)
        y_tr = np.array([label_to_idx[r["best_strategy"]] for r in train])
        X_te_num = np.array([[r[f] for f in feature_names] for r in test], dtype=float)

        # Scale numeric features
        scaler = StandardScaler().fit(X_tr_num)
        X_tr_num = scaler.transform(X_tr_num)
        X_te_num = scaler.transform(X_te_num)

        # Add TF-IDF features from question text
        try:
            tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2),
                                    min_df=2, stop_words="english").fit(
                [r["_question"] for r in train])
            X_tr_tfidf = tfidf.transform([r["_question"] for r in train])
            X_te_tfidf = tfidf.transform([r["_question"] for r in test])
            X_tr = hstack([csr_matrix(X_tr_num), X_tr_tfidf])
            X_te = hstack([csr_matrix(X_te_num), X_te_tfidf])
        except ValueError:
            X_tr, X_te = X_tr_num, X_te_num

        # LR classifier
        try:
            lr = LogisticRegression(max_iter=5000, class_weight="balanced",
                                    C=0.5, solver="liblinear").fit(X_tr, y_tr)
            lr_preds = lr.predict(X_te)
        except ValueError:
            lr_preds = np.full(len(test), label_to_idx["self_consistency"])

        # GB classifier (on dense features only, since sparse is slow)
        try:
            X_tr_dense = X_tr.toarray() if hasattr(X_tr, "toarray") else X_tr
            X_te_dense = X_te.toarray() if hasattr(X_te, "toarray") else X_te
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=3).fit(X_tr_dense, y_tr)
            gb_preds = gb.predict(X_te_dense)
        except ValueError:
            gb_preds = np.full(len(test), label_to_idx["self_consistency"])

        # Evaluate: for each test example, route to predicted strategy.
        # If predicted = "none" (no strategy works), fall back to SC.
        def eval_router(preds):
            correct = 0
            total_tokens = 0
            for j, r in enumerate(test):
                s_idx = int(preds[j])
                s = strat_labels[s_idx]
                if s == "none":
                    s = "self_consistency"
                ok_key = {
                    "standard_cot": "cot_correct",
                    "self_consistency": "sc_correct",
                    "backward_cloze": "bc_correct",
                }[s]
                correct += r[ok_key]
                # Token cost from cell data
                cell = cells[(r["model"], r["benchmark"])]
                total_tokens += cell["per_strategy_mean_tokens"].get(s, 0)
            return correct / max(len(test), 1), total_tokens / max(len(test), 1)

        lr_acc, lr_tok = eval_router(lr_preds)
        gb_acc, gb_tok = eval_router(gb_preds)

        # Baselines
        sc_acc = np.mean([r["sc_correct"] for r in test])
        oracle_acc = np.mean([r["any_correct"] for r in test])
        sc_tok = np.mean([cells[(r["model"], r["benchmark"])]
                         ["per_strategy_mean_tokens"]["self_consistency"]
                         for r in test])

        gap = oracle_acc - sc_acc
        cap_lr = (lr_acc - sc_acc) / gap if gap > 0 else 0
        cap_gb = (gb_acc - sc_acc) / gap if gap > 0 else 0
        cost_lr = lr_tok / sc_tok if sc_tok > 0 else 1.0

        print(f"{test_b:<15} {sc_acc:>6.3f} {oracle_acc:>7.3f} "
              f"{lr_acc:>6.3f} {gb_acc:>6.3f} "
              f"{cap_lr:>+7.2f} {cap_gb:>+7.2f} {cost_lr:>7.2f}x")
        summary.append({
            "test_benchmark": test_b,
            "sc_acc": sc_acc,
            "oracle_acc": oracle_acc,
            "router_lr_acc": lr_acc,
            "router_gb_acc": gb_acc,
            "router_lr_tokens": lr_tok,
            "sc_tokens": sc_tok,
            "cap_lr": cap_lr,
            "cap_gb": cap_gb,
            "compute_ratio_lr": cost_lr,
        })

    avg_cap_lr = np.mean([s["cap_lr"] for s in summary])
    avg_cap_gb = np.mean([s["cap_gb"] for s in summary])
    avg_cost_lr = np.mean([s["compute_ratio_lr"] for s in summary])
    print(f"\n{'Average':<15} {'':<6} {'':<7} {'':<6} {'':<6} "
          f"{avg_cap_lr:>+7.2f} {avg_cap_gb:>+7.2f} {avg_cost_lr:>7.2f}x")

    max_cap = max(avg_cap_lr, avg_cap_gb)
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if max_cap >= 0.40 and avg_cost_lr < 0.5:
        print(f">>> STRONG POSITIVE: router captures {max_cap:.0%} of gap at "
              f"{avg_cost_lr:.0%} of SC compute.")
        print("    Proceed with method paper.")
    elif max_cap >= 0.20:
        print(f">>> MARGINAL: router captures {max_cap:.0%} of gap.")
        print("    Need more features or data.")
    else:
        print(f">>> NEGATIVE: router captures only {max_cap:.0%} → question "
              "text alone doesn't carry signal. Dead.")

    # Save
    Path("results/question_router").mkdir(parents=True, exist_ok=True)
    with open("results/question_router/router_summary.json", "w") as f:
        json.dump({
            "summary": summary,
            "avg_cap_lr": avg_cap_lr,
            "avg_cap_gb": avg_cap_gb,
            "avg_cost_lr": avg_cost_lr,
            "verdict": ("positive" if (max_cap >= 0.40 and avg_cost_lr < 0.5) else
                       ("marginal" if max_cap >= 0.20 else "negative")),
            "n_train_instances": len(data_rows),
            "best_strategy_distribution": dict(counts),
        }, f, indent=2, default=str)
    print(f"\nSaved: results/question_router/router_summary.json")


if __name__ == "__main__":
    main()
