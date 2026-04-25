#!/usr/bin/env python3
"""Create fixed calibration/test split manifests.

GPT-5.5 Pro Task 5: prevent first-N bias and train/test leakage.
Creates seeded random splits saved as JSON manifests.

Usage:
    python code/split_manifest.py \
        --benchmarks gsm8k,math_hard,strategyqa,arc_challenge,bbh_logic \
        --n_calib 50 --n_test 100 --seed 11 \
        --output configs/splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def stable_shuffle(items, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(items))
    return [items[i] for i in indices]


def create_manifest(benchmark, n_calib, n_test, seed=11):
    from benchmarks import load_benchmark, load_math, load_bbh

    if benchmark == "math_hard":
        examples = load_math(max_examples=None, levels=[4, 5])
    elif benchmark == "bbh_logic":
        examples = load_bbh(subtasks=["logical_deduction_five_objects"])
    else:
        examples = load_benchmark(benchmark, max_examples=None)

    all_ids = [ex.example_id for ex in examples]
    shuffled = stable_shuffle(all_ids, seed)

    # Capture data-source metadata so future loads can detect drift.
    src_meta = {}
    if examples:
        md0 = getattr(examples[0], "metadata", {}) or {}
        src_meta = {
            "source_repo": md0.get("source_repo", ""),
            "source_split": md0.get("source_split", ""),
            "dataset_revision": md0.get("dataset_revision", ""),
        }

    n_total = min(n_calib + n_test, len(shuffled))
    n_c = min(n_calib, n_total)
    n_t = min(n_test, n_total - n_c)

    calib_ids = sorted(shuffled[:n_c])
    test_ids = sorted(shuffled[n_c:n_c + n_t])

    # Fingerprint includes source metadata so a dataset-revision change
    # produces a different fingerprint and is auto-detectable.
    id_hash = hashlib.sha256(
        json.dumps(sorted(all_ids)).encode()
    ).hexdigest()[:16]
    fingerprint = hashlib.sha256(
        json.dumps({
            "benchmark": benchmark, "seed": seed,
            "n_total": len(all_ids), "id_hash": id_hash,
            **src_meta,
        }, sort_keys=True).encode()
    ).hexdigest()[:16]

    return {
        "benchmark": benchmark,
        "seed": seed,
        "total_available": len(all_ids),
        "n_calib": len(calib_ids),
        "n_test": len(test_ids),
        "fingerprint": fingerprint,
        "id_hash": id_hash,
        "source_repo": src_meta.get("source_repo", ""),
        "source_split": src_meta.get("source_split", ""),
        "dataset_revision": src_meta.get("dataset_revision", ""),
        "calib_ids": calib_ids,
        "test_ids": test_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=str,
                        default="gsm8k,math_hard,strategyqa,arc_challenge,bbh_logic")
    parser.add_argument("--n_calib", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=str, default="configs/splits")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.check:
        ok = True
        files = [f for f in os.listdir(args.output) if f.endswith(".json")]
        if not files:
            print("FAIL: no manifests in directory — empty tree is NOT 'valid'.")
            sys.exit(2)
        for f in files:
            data = json.load(open(os.path.join(args.output, f)))
            overlap = set(data["calib_ids"]) & set(data["test_ids"])
            if overlap:
                print(f"FAIL: {f} has {len(overlap)} overlapping IDs")
                ok = False
            elif not data.get("calib_ids") or not data.get("test_ids"):
                print(f"FAIL: {f} has empty calib_ids or test_ids")
                ok = False
            else:
                print(f"OK: {f} — calib={data['n_calib']}, test={data['n_test']}, "
                      f"src={data.get('source_repo', '?')}, no overlap")
        if ok:
            print("All manifests valid.")
        else:
            sys.exit(2)
        return

    for bname in args.benchmarks.split(","):
        bname = bname.strip()
        try:
            manifest = create_manifest(bname, args.n_calib, args.n_test, args.seed)
            out_path = os.path.join(args.output, f"{bname}.json")
            with open(out_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"[OK] {bname}: calib={manifest['n_calib']}, test={manifest['n_test']}, "
                  f"total={manifest['total_available']}, fingerprint={manifest['fingerprint']}")
        except Exception as e:
            print(f"[SKIP] {bname}: {e}")


if __name__ == "__main__":
    main()
