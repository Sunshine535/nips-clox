# CLOX v2 — 拓扑感知的计算最优推理策略选择

## 项目简介

CLOX 提出了一种基于推理 trace 拓扑结构的自适应推理策略选择方法。核心发现：推理 trace 具有可测量的结构属性——**局部可恢复性 (r̄)** 和**误差传播长度 (ℓ)**——这两个指标决定了哪种推理策略最优。CLOX-Adaptive 从少量 pilot traces 估算拓扑，自动路由到最优策略，在匹配 Self-Consistency 准确率的同时节省 40-60% 计算开销。

**Review 状态**: Round 3, Score 6.5/10

## 环境安装

```bash
cd /workspace/nips-clox
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install vllm transformers datasets numpy scipy matplotlib sentencepiece
```

## 快速开始（Smoke Test）

```bash
source venv/bin/activate
cd code
# 合成 DAG 验证（无需 GPU，~1 分钟）
python3 synthetic_dag.py

# 带模型的 smoke test（需要 GPU）
python3 run_clox.py --model Qwen/Qwen2.5-7B-Instruct --tp 1 \
    --benchmarks gsm8k --phase topology --max_examples 10
```

## 完整实验流程

### Phase 1: 拓扑特征化
```bash
# 2×GPU, TP=2, 200 examples per benchmark
CUDA_VISIBLE_DEVICES=0,1 python3 run_full_experiment.py \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ --tp 2 \
    --phase topology --benchmarks gsm8k,math,strategyqa,arc_challenge \
    --output ../results/v5
```

### Phase 2: 策略对比（9 策略 × 3 seeds × 4 benchmarks）
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 run_full_experiment.py \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ --tp 2 \
    --phase strategies --seeds 11,23,37 \
    --output ../results/v5
```

### Phase 3: Proxy 验证
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 run_full_experiment.py \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ --tp 2 \
    --phase proxy --output ../results/v5
```

### 全流程一键运行
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 run_full_experiment.py \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ --tp 2 \
    --phase all --seeds 11,23,37 --output ../results/v5
```

### 多卡配置
- `--tp 2`: 2 卡 Tensor Parallel（32B 模型必须）
- `--tp 1`: 单卡（8B 以下模型）
- 可同时跑两个模型：GPU 0,1 跑 32B (TP=2), GPU 2,3 跑 8B (TP=2)

## 断点续训

`run_full_experiment.py` 使用 checkpoint 文件（`.ckpt_*.json`）实现断点续训：
- 每 50 个 example 自动保存进度
- 重新运行同一命令会跳过已完成的 (strategy, seed) 组合
- 强制重跑：删除对应 benchmark 目录下的 `.ckpt_*.json` 文件

## 已有结果

| 数据 | 位置 | 状态 |
|------|------|------|
| 合成 DAG 验证 | `results/synthetic/` | 完成，理论预测准确率 83% |
| 32B-AWQ 拓扑 (4 benchmarks × 200) | `results/v3/Qwen2.5-32B-Instruct-AWQ/` | 完成 |
| 8B Pilot (4 benchmarks × 50) | `results/v4/Qwen3-8B/pilot/` | 完成 |
| 8B 拓扑 (3 benchmarks × 200) | `results/v4/Qwen3-8B/` | 完成 |
| 32B GSM8K 策略 (部分) | `results/v3/` | 20/27 combos |

**关键数字** (32B GSM8K):
- Targeted Repair: **96.34%** >> SC: 91.11% >> CoT: 89.33% (p<0.001)

## 项目结构

```
code/
  engine.py              # vLLM 生成引擎 (TP 支持)
  strategies_v2.py       # 9 个推理策略 + STRATEGY_REGISTRY
  topology_v2.py         # 拓扑估算 (r̄, ℓ)
  run_clox.py            # 主 runner (分 phase)
  run_full_experiment.py # 全流程 runner (checkpoint 支持)
  synthetic_dag.py       # 合成 DAG 理论验证
  benchmarks.py          # 4 benchmark 加载器
  evaluation.py          # Bootstrap CI, McNemar, Cohen's d
  analyze_v2.py          # 结果分析 + 绘图
results/
  synthetic/             # DAG 验证结果
  v3/                    # 32B-AWQ 结果
  v4/                    # 8B 结果
paper/                   # LaTeX 论文草稿
```

## 下一步（Reviewer 要求）

1. 完成 32B 全部 9 策略 × 3 seeds × 4 benchmarks
2. 评估 CLOX-Adaptive 在 held-out split 上的表现
3. Proxy validation（3/5/8 pilot traces vs 30 ground truth）
4. 完整统计检验（paired bootstrap CI, McNemar, Bonferroni）
5. 论文重写，纳入 short-ℓ regime 发现
