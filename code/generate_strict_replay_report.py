#!/usr/bin/env python3
"""Deterministic markdown generator for the strict-replay report.

GPT-5.5 Pro Round-5 Task 1: ensure `reports/STRICT_REPLAY_RESULTS.md` matches
the strict-replay JSON files exactly. The previous markdown was hand-written
and drifted from the actual numbers. This script (a) regenerates the report
from the JSON inputs and (b) provides `--check` mode that compares the
markdown back against the JSON and exits non-zero on any disagreement >0.001.

Usage:
    # regenerate
    python code/generate_strict_replay_report.py \
        --inputs results/agd/Qwen3.5-27B/agd_results_strict.json,\
results/agd/Qwen3.5-9B/agd_results_strict.json \
        --out reports/STRICT_REPLAY_RESULTS.md

    # consistency check
    python code/generate_strict_replay_report.py \
        --check reports/STRICT_REPLAY_RESULTS.md \
        --inputs results/agd/Qwen3.5-27B/agd_results_strict.json,\
results/agd/Qwen3.5-9B/agd_results_strict.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Iterable

THRESHOLDS = ["0.5", "0.75", "1.0"]


def _model_label(path: str) -> str:
    """Extract a model label from the path."""
    parts = os.path.normpath(path).split(os.sep)
    for p in parts[::-1]:
        if "Qwen" in p or "qwen" in p.lower():
            return p
    return os.path.basename(os.path.dirname(path)) or "unknown"


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _benchmark_row(name: str, payload: dict) -> str:
    """One row of the per-benchmark table."""
    sc_legacy = payload["sc8_legacy_acc"]
    sc_strict = payload["sc8_strict_acc"]
    delta_sc = payload["delta_sc8"]
    agd_l = payload["agd_legacy_acc"]["0.5"]
    agd_s = payload["agd_strict_acc"]["0.5"]
    return (
        f"| {name} | {payload['answer_type']} "
        f"| {_fmt(sc_legacy)} | {_fmt(sc_strict)} | {_fmt(delta_sc)} "
        f"| {_fmt(agd_l)} | {_fmt(agd_s)} | {_fmt(agd_s - sc_strict)} |"
    )


def _section(model_label: str, data: dict) -> list[str]:
    out = [f"## {model_label} AGD", ""]
    out.append(
        "| Benchmark | Type | SC(8) legacy | SC(8) strict | Δ SC | "
        "AGD(0.5) legacy | AGD(0.5) strict | AGD-vs-SC strict |"
    )
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    benchmarks = data["benchmarks"]
    sums_legacy = sums_strict = agd_legacy_sum = agd_strict_sum = 0.0
    for bname, payload in benchmarks.items():
        out.append(_benchmark_row(bname, payload))
        sums_legacy += payload["sc8_legacy_acc"]
        sums_strict += payload["sc8_strict_acc"]
        agd_legacy_sum += payload["agd_legacy_acc"]["0.5"]
        agd_strict_sum += payload["agd_strict_acc"]["0.5"]
    n = len(benchmarks) or 1
    mean_sc_l = sums_legacy / n
    mean_sc_s = sums_strict / n
    mean_agd_l = agd_legacy_sum / n
    mean_agd_s = agd_strict_sum / n
    out.append(
        f"| **mean** | | **{_fmt(mean_sc_l)}** | **{_fmt(mean_sc_s)}** "
        f"| **{_fmt(mean_sc_s - mean_sc_l)}** "
        f"| **{_fmt(mean_agd_l)}** | **{_fmt(mean_agd_s)}** "
        f"| **{_fmt(mean_agd_s - mean_sc_s)}** |"
    )
    out.append("")
    return out


def _findings(sections_data: list[tuple[str, dict]]) -> list[str]:
    out = ["## Findings", ""]
    for label, data in sections_data:
        bench = data["benchmarks"]
        # 27B/9B style summary
        positives = []
        negatives = []
        for bname, payload in bench.items():
            d = payload["agd_strict_acc"]["0.5"] - payload["sc8_strict_acc"]
            (positives if d >= 0 else negatives).append((bname, d))
        positives.sort(key=lambda kv: kv[1], reverse=True)
        negatives.sort(key=lambda kv: kv[1])
        line_parts = [f"**{label}**:"]
        for bname, d in positives:
            sign = "+" if d > 0 else "±"
            line_parts.append(f"{bname} {sign}{abs(d):.3f}")
        for bname, d in negatives:
            line_parts.append(f"{bname} -{abs(d):.3f}")
        out.append("- " + "; ".join([line_parts[0]] + line_parts[1:]))
    out.append("")
    out.append(
        "Strict replay weakens the legacy 'AGD universally non-negative on 27B' "
        "claim: under strict metric the headline is reduced to "
        "'AGD non-negative on a subset of benchmarks; 9B remains capability-limited'."
    )
    out.append("")
    return out


def _generate_markdown(jsons: list[dict], paths: list[str]) -> str:
    out: list[str] = [
        "# Strict Metric Replay Results",
        "",
        ("> AUTO-GENERATED from `code/generate_strict_replay_report.py` over "
         "`results/agd/*/agd_results_strict.json`. Do not edit by hand. "
         "Run `python code/generate_strict_replay_report.py --check ...` "
         "as part of CI to verify markdown ↔ JSON consistency."),
        "",
        ("GPT-5.5 Pro Round 4 review identified that legacy "
         "`evaluation.check_answer` contained a substring fallback that "
         "contaminates accuracy on text/MC/boolean answers. This report "
         "shows what AGD numbers look like when re-scored through "
         "`check_answer_strict`."),
        "",
        "Source script: `code/replay_results_strict.py` (committed).",
        "",
    ]
    sections_data: list[tuple[str, dict]] = []
    for json_path, data in zip(paths, jsons):
        label = _model_label(json_path)
        out.extend(_section(label, data))
        out.append(f"Source file: `{json_path}`.")
        out.append("")
        sections_data.append((label, data))
    out.extend(_findings(sections_data))
    out.append("## Reproduction")
    out.append("")
    out.append("```bash")
    for p in paths:
        # legacy path = replace _strict.json with .json
        legacy = p.replace("_strict.json", ".json")
        out.append(
            f"python code/replay_results_strict.py --input {legacy} --out {p}"
        )
    join = ",".join(paths)
    out.append(
        f"python code/generate_strict_replay_report.py "
        f"--inputs {join} --out reports/STRICT_REPLAY_RESULTS.md"
    )
    out.append(
        f"python code/generate_strict_replay_report.py "
        f"--check reports/STRICT_REPLAY_RESULTS.md --inputs {join}"
    )
    out.append("```")
    out.append("")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────
# Consistency check — extract numbers from markdown, compare to JSON


_NUM_RE = re.compile(r"-?\d+\.\d{3}")


def _extract_table_rows(markdown: str) -> dict[tuple[str, str], list[float]]:
    """Pull each table row's numeric cells, keyed by (section_label, row_label)."""
    result: dict[tuple[str, str], list[float]] = {}
    current_section = ""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("## "):
            heading = line[3:].strip()
            heading = re.sub(r"\s+AGD\s*$", "", heading)
            current_section = heading
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0] in ("Benchmark", "---", ""):
            continue
        label = cells[0].replace("**", "").strip()
        nums = [float(m) for m in _NUM_RE.findall(line)]
        if nums:
            result[(current_section, label)] = nums
    return result


def check_consistency(markdown: str, jsons: list[dict], paths: list[str],
                      tol: float = 0.001) -> list[str]:
    """Return human-readable mismatch strings; empty list = consistent."""
    mismatches: list[str] = []
    rows = _extract_table_rows(markdown)
    for json_path, data in zip(paths, jsons):
        section = _model_label(json_path)
        bench = data["benchmarks"]
        for bname, payload in bench.items():
            expected = [
                payload["sc8_legacy_acc"],
                payload["sc8_strict_acc"],
                payload["delta_sc8"],
                payload["agd_legacy_acc"]["0.5"],
                payload["agd_strict_acc"]["0.5"],
                payload["agd_strict_acc"]["0.5"] - payload["sc8_strict_acc"],
            ]
            key = (section, bname)
            actual = rows.get(key, [])
            if not actual:
                mismatches.append(
                    f"{json_path} :: section='{section}' :: {bname} — markdown row missing"
                )
                continue
            if len(actual) < len(expected):
                mismatches.append(
                    f"{json_path} :: section='{section}' :: {bname} — "
                    f"markdown has {len(actual)} numeric cells, expected {len(expected)}"
                )
                continue
            for i, (a, e) in enumerate(zip(actual[: len(expected)], expected)):
                if abs(a - e) > tol:
                    mismatches.append(
                        f"{json_path} :: section='{section}' :: {bname} col[{i}] "
                        f"markdown={a:.3f} json={e:.3f} Δ={abs(a-e):.4f}"
                    )
    return mismatches


# ──────────────────────────────────────────────────────────────────


def _load_inputs(inputs_arg: str) -> tuple[list[dict], list[str]]:
    paths = [p.strip() for p in inputs_arg.split(",") if p.strip()]
    jsons = [json.load(open(p)) for p in paths]
    return jsons, paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True,
                   help="Comma-separated paths to *_strict.json files.")
    p.add_argument("--out", default="",
                   help="Markdown output path (regen mode).")
    p.add_argument("--check", default="",
                   help="Markdown path to verify against JSON (check mode).")
    args = p.parse_args()

    jsons, paths = _load_inputs(args.inputs)

    if args.check:
        if not os.path.exists(args.check):
            print(f"FAIL: markdown not found: {args.check}", file=sys.stderr)
            sys.exit(2)
        markdown = open(args.check).read()
        mismatches = check_consistency(markdown, jsons, paths)
        if mismatches:
            print("FAIL: markdown ↔ JSON mismatches:")
            for m in mismatches:
                print(f"  {m}")
            sys.exit(2)
        print(f"OK: {args.check} matches all JSON inputs (tol=0.001).")
        return

    if not args.out:
        print("FAIL: must pass --out or --check", file=sys.stderr)
        sys.exit(2)

    md = _generate_markdown(jsons, paths)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    # Self-verify
    mismatches = check_consistency(md, jsons, paths)
    if mismatches:
        print("FAIL: regenerated markdown does not match its own JSON sources!")
        for m in mismatches:
            print(f"  {m}")
        sys.exit(2)
    print(f"OK: regenerated → {args.out}")


if __name__ == "__main__":
    main()
