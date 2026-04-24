"""Immutable result schema for PCS experiments.

GPT-5.5 Pro Task 2: stop losing raw evidence.
Every run saves manifest + per-example JSONL. Never deletes raw data.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
    except Exception:
        return "unknown"


def create_run_manifest(
    output_dir: str,
    model: str,
    benchmark: str,
    split: str,
    seed: int,
    config: dict,
    command: str = "",
) -> dict:
    manifest = {
        "timestamp": datetime.utcnow().isoformat(),
        "git_hash": get_git_hash(),
        "model": model,
        "benchmark": benchmark,
        "split": split,
        "seed": seed,
        "config": config,
        "command": command,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def save_per_example(output_dir: str, strategy: str, seed: int, rows: list[dict]):
    per_ex_dir = os.path.join(output_dir, "per_example")
    os.makedirs(per_ex_dir, exist_ok=True)
    path = os.path.join(per_ex_dir, f"{strategy}_s{seed}.jsonl")
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def load_per_example(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_candidate_outputs(output_dir: str, example_id: str, candidates: list[dict]):
    cand_dir = os.path.join(output_dir, "candidate_outputs")
    os.makedirs(cand_dir, exist_ok=True)
    path = os.path.join(cand_dir, f"{example_id}.jsonl")
    with open(path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, default=str) + "\n")
