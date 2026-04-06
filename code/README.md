# Code Package for CLOX: Partial Masking as a Control for Synthetic Reasoning

## Description
Experiment code for topology-dependent inference-time strategy selection.

## Project Files
- `main.py` — Experiment entry point (checkpoint + multi-GPU)
- `strategies.py` — 8 inference strategies (CoT, SC, UTMR, CLOX-Adaptive, ...)
- `topology.py` — EPL + recoverability estimation from pilot traces
- `evaluation.py` — Bootstrap CI, McNemar, Cohen's d, Bonferroni
- `benchmarks.py` — GSM8K, MATH, StrategyQA, ARC-C, BBH loaders

## How to Run
```bash
pip install -r requirements.txt
python main.py --model Qwen/Qwen2.5-7B-Instruct --benchmarks gsm8k --strategies all --seeds 11
```

## Dependencies
Install dependencies with `pip install -r requirements.txt`.
