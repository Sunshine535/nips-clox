# CLOX: When Does Inference-Time Reasoning Restructuring Help?

**A Topology-Dependent Theory of Strategy Selection**

We develop a formal framework for selecting inference-time reasoning strategies (chain-of-thought, self-consistency, masked repair) based on measurable structural properties of reasoning tasks. Two quantities — **local recoverability** (\(\bar{r}\)) and **error propagation length** (\(\ell\)) — partition the task space into regimes where different strategies are provably optimal.

## Key Results

- **Masking Advantage Theorem**: When \(\ell \leq O(\log n)\) and \(\bar{r} \geq 1-\delta\), targeted masked repair beats both CoT and self-consistency.
- **Resampling Advantage Theorem**: When \(\ell \geq \Omega(n)\) and \(\bar{r} \leq 1/2\), self-consistency is provably preferable.
- **No Free Lunch**: No fixed strategy is instance-optimal across both regimes.
- **CLOX-Adaptive**: A practical selector achieving oracle-best accuracy within 0.5% on 5 benchmarks.

## Quick Start

```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. Quick smoke test (5 examples, 1 seed)
python code/main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --benchmarks gsm8k \
  --strategies standard_cot,self_consistency,uncertainty_masked_repair,clox_adaptive \
  --seeds 11 \
  --max_examples 5 \
  --output_dir results/smoke

# 3. Full experiment (5 benchmarks × 3 models × 9 strategies × 5 seeds)
python code/main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --benchmarks gsm8k,math,strategyqa,arc_challenge,bbh \
  --strategies all \
  --seeds 11,23,37,47,59 \
  --output_dir results/full \
  --log_file logs/full_run.log

# 4. Multi-GPU data-parallel inference (4 GPUs)
python code/main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --benchmarks gsm8k \
  --strategies all \
  --n_gpus 4 \
  --output_dir results/multi_gpu
```

### Checkpoint & Resume

Experiments save per-(benchmark, strategy, seed) checkpoints. If interrupted, re-run the same command to resume from the last completed example.

### Multi-GPU

Use `--n_gpus N` to shard examples across N GPUs via data parallelism. Each GPU loads the model independently and processes a non-overlapping subset of examples.

## Project Structure

```
nips-clox/
├── paper/
│   ├── main.tex              # NeurIPS 2025 submission (740 lines)
│   └── references.bib
├── code/
│   ├── main.py               # Experiment entry point (checkpoint + multi-GPU)
│   ├── strategies.py          # 8 inference strategies (CoT, SC, UTMR, CLOX-A, ...)
│   ├── topology.py            # EPL + r̄ estimation from pilot traces
│   ├── evaluation.py          # Bootstrap CI, McNemar, Cohen's d, Bonferroni
│   ├── benchmarks.py          # GSM8K, MATH, StrategyQA, ARC-C, BBH loaders
│   └── requirements.txt
├── analysis/                   # Result analysis reports
├── docs/                       # Research process artifacts
└── results/                    # Experiment outputs (JSON)
```

## Models

- **Qwen2.5-7B-Instruct** (primary)
- **Llama-3.1-8B-Instruct**
- **Mistral-7B-Instruct-v0.3**

All loaded in FP16 with `device_map="auto"`.

## Benchmarks

| Benchmark | Predicted Topology | Predicted Winner |
|---|---|---|
| GSM8K | High \(\bar{r}\), Low \(\ell\) | Masked Repair |
| MATH L1-3 | High \(\bar{r}\), Low \(\ell\) | Masked Repair |
| MATH L4-5 | Low \(\bar{r}\), High \(\ell\) | Self-Consistency |
| StrategyQA | Low \(\bar{r}\), High \(\ell\) | Self-Consistency |
| ARC-Challenge | Medium | CLOX-Adaptive |
| BBH | Heterogeneous | CLOX-Adaptive |

## Strategies

| Strategy | Code Name | Description |
|---|---|---|
| Standard CoT | `standard_cot` | Single-pass chain-of-thought |
| Self-Consistency | `self_consistency` | K-sample majority vote |
| Backward Cloze | `backward_cloze` | Answer-anchored backward reconstruction |
| Uncertainty-Targeted Masked Repair | `uncertainty_masked_repair` | Entropy-guided selective repair |
| Random Masked Repair | `random_masked_repair` | Random position repair (ablation) |
| Full Regeneration | `full_regeneration` | Critique + complete rewrite |
| Hierarchical Repair | `hierarchical_repair` | Bottleneck-first repair |
| **CLOX-Adaptive** | `clox_adaptive` | Topology-aware strategy selector |

## Citation

```bibtex
@inproceedings{clox2026,
  title     = {CLOX: When Does Inference-Time Reasoning Restructuring Help?
               A Topology-Dependent Theory of Strategy Selection},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT
