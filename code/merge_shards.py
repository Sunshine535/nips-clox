#!/usr/bin/env python3
"""Merge pilot results from multiple shards into one pilot_results.json.

Usage: python3 code/merge_shards.py results/pilot
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path


def merge(base_dir: str) -> None:
    base = Path(base_dir)
    shard_dirs = sorted([d for d in base.glob("shard_*") if d.is_dir()])
    if not shard_dirs:
        print(f"No shard_* dirs found in {base_dir}")
        sys.exit(1)

    print(f"Found {len(shard_dirs)} shards: {[d.name for d in shard_dirs]}")

    merged_results: dict[str, dict] = {}
    merged_topology: dict[str, dict] = {}
    merged_problems: list = []
    config = None

    for d in shard_dirs:
        res_path = d / "pilot_results.json"
        if not res_path.exists():
            print(f"  [SKIP] {d.name}: no pilot_results.json")
            continue
        with open(res_path) as f:
            data = json.load(f)

        if config is None:
            config = data["config"]
        merged_problems.extend(data.get("problems", []))
        merged_topology.update(data.get("topology", {}))
        for sname, per_example in data.get("strategy_results", {}).items():
            if sname not in merged_results:
                merged_results[sname] = {}
            merged_results[sname].update(per_example)
        n_done = sum(len(v) for v in data.get("strategy_results", {}).values())
        print(f"  [OK]   {d.name}: {len(data.get('problems', []))} problems, "
              f"{len(data.get('strategy_results', {}))} strategies, {n_done} entries")

    # De-dup problems by id (shards select non-overlapping subsets already)
    seen_ids = set()
    dedup_problems = []
    for p in merged_problems:
        if p["id"] not in seen_ids:
            seen_ids.add(p["id"])
            dedup_problems.append(p)

    output = {
        "config": config,
        "problems": dedup_problems,
        "topology": merged_topology,
        "strategy_results": merged_results,
    }
    out_path = base / "pilot_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    n_problems = len(dedup_problems)
    print(f"\n=== MERGED ===")
    print(f"Total problems: {n_problems}")
    print(f"Strategies: {list(merged_results.keys())}")
    for sname, per_ex in merged_results.items():
        correct = sum(1 for v in per_ex.values() if v.get("correct"))
        n = len(per_ex)
        print(f"  {sname:25s}  {correct}/{n}  acc={correct/max(n,1):.1%}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 merge_shards.py <pilot_dir>")
        sys.exit(1)
    merge(sys.argv[1])
