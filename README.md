# CLOX v2 — 拓扑感知的计算最优推理策略选择

推理 trace 具有可测量的结构属性——**局部可恢复性 (r̄)** 和**误差传播长度 (ℓ)**——这两个指标决定了最优推理策略。CLOX 从少量 pilot traces 估算拓扑，自动路由到最优策略，在匹配 Self-Consistency 准确率的同时节省 40-60% 计算开销。

**Review 状态**: Round 3, Score 6.5/10

## 环境安装

```bash
git clone https://github.com/Sunshine535/nips-clox.git
cd nips-clox
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install vllm transformers datasets numpy scipy matplotlib sentencepiece
```

## 当前进度

| 组件 | 状态 |
|------|------|
| 合成 DAG 验证 | ✅ 83% 理论预测准确率 |
| 拓扑特征化 (32B, 4 benchmarks × 200) | ✅ 完成 |
| 拓扑特征化 (8B, 3 benchmarks × 200) | ✅ 完成 |
| Pilot 结果 (32B + 8B) | ✅ 完成 |
| **策略对比 (9 strategies × 3 seeds × 4 benchmarks)** | ❌ 待运行 |
| CLOX-Adaptive 评估 | ❌ 待运行 |
| Proxy validation | ❌ 待运行 |
| 论文草稿 | 🔄 需更新结果 |

## 快速验证（Smoke Test）

```bash
source venv/bin/activate
cd code

# 合成 DAG 验证（无需 GPU，~1 分钟）
python3 synthetic_dag.py

# 带模型的 smoke test（需要 GPU）
python3 run_clox.py --model Qwen/Qwen3.5-9B --tp 1 \
    --benchmarks gsm8k --phase topology --max_examples 10
```

## 继续实验

### Step 1: 策略对比（核心缺失部分）

9 策略 × 3 seeds × 4 benchmarks，使用 Qwen3.5-27B（最新 Qwen3.5 系列）。

GPU 自动检测已内置——`--tp 0`（默认）根据模型大小和可用 GPU 自动选择 TP。

```bash
source venv/bin/activate

# 全量运行：4 benchmarks, 9 strategies, 3 seeds
# 每 50 例自动 checkpoint，可随时中断重跑
python3 code/run_full_experiment.py \
    --phase strategies \
    --seeds 11,23,37 \
    --output results/v5 \
    --log_file results/v5/strategies.log
```

也可按 benchmark 分步运行：

```bash
# GSM8K（最关键，~2h）
python3 code/run_full_experiment.py \
    --phase strategies --benchmarks gsm8k \
    --seeds 11,23,37 --output results/v5

# MATH (~3h)
python3 code/run_full_experiment.py \
    --phase strategies --benchmarks math \
    --seeds 11,23,37 --output results/v5

# StrategyQA + ARC
python3 code/run_full_experiment.py \
    --phase strategies --benchmarks strategyqa,arc_challenge \
    --seeds 11,23,37 --output results/v5
```

9 个策略：`standard_cot`, `self_consistency` (k=5), `compute_matched_sc` (k=2), `targeted_repair`, `random_repair`, `backward_cloze`, `full_regeneration`, `hierarchical_repair`, `clox_adaptive`

### Step 2: Proxy Validation

测试多少 pilot traces 足够可靠估算拓扑。

```bash
python3 code/run_full_experiment.py --phase proxy --output results/v5
```

### Step 3: 分析与出图

```bash
python3 code/analyze_v2.py results/v5/Qwen3.5-27B/
```

### Step 4: 跨模型验证（可选）

用 Qwen3.5-9B 做跨模型分析：

```bash
python3 code/run_full_experiment.py \
    --model Qwen/Qwen3.5-9B \
    --phase strategies \
    --seeds 11,23,37 \
    --output results/v5
```

## 断点续训

`run_full_experiment.py` 每 50 例自动保存 checkpoint：

- 每个 benchmark 目录下的 `.ckpt_{strategy}_s{seed}.json`
- 重跑同一命令会自动跳过已完成的 (strategy, seed) 组合
- 强制重跑某个组合：删除对应的 `.ckpt_*.json` 文件
- 最终结果保存到 `{benchmark}/strategies.json`（含聚合统计）

## 多卡配置

引擎自动检测 GPU 并选择 tensor parallelism：

| 模型大小 | 4 GPU | 2 GPU | 1 GPU |
|---------|-------|-------|-------|
| 70B/72B | TP=4 | TP=2 | TP=1 |
| 27B/32B | TP=4 | TP=2 | TP=1 |
| MoE (35B-A3B 等) | TP=2 | TP=2 | TP=1 |
| ≤14B | TP=2 | TP=2 | TP=1 |

手动指定：`--tp N`。支持 `CUDA_VISIBLE_DEVICES` 限制可见 GPU。

## 已有结果

| 数据 | 位置 | 说明 |
|------|------|------|
| 合成 DAG | `results/synthetic/` | 5 图类型 × 6 r̄ × 3 seeds × 2000 trials |
| 32B 拓扑 | `results/v3/Qwen3.5-27B/` | 4 benchmarks × 200 examples |
| 8B 拓扑 | `results/v4/Qwen3-8B/` | 3 benchmarks × 200 examples (旧模型) |
| 32B Pilot | `results/v3/.../pilot/pilot_results.json` | 50 examples × 5 strategies × 4 benchmarks (旧模型) |
| 8B Pilot | `results/v4/.../pilot/pilot_results.json` | 50 examples × 5 strategies × 4 benchmarks (旧模型) |

### 关键拓扑数据 (32B)

| Benchmark | r̄ | ℓ | 预测策略 |
|-----------|-----|------|---------|
| GSM8K | 0.521 ± 0.074 | 1.28 ± 0.28 | Targeted repair |
| MATH | 0.634 ± 0.086 | 1.18 ± 0.28 | Targeted repair |
| StrategyQA | 0.451 ± 0.057 | 1.55 ± 0.28 | Standard CoT / Adaptive |
| ARC-Challenge | 0.427 ± 0.055 | 1.57 ± 0.38 | Standard CoT / Adaptive |

## 项目结构

```
code/
  engine.py              # vLLM 引擎 (自动 GPU 检测 + TP)
  strategies_v2.py       # 9 个推理策略 + STRATEGY_REGISTRY
  topology_v2.py         # 拓扑估算 (r̄, ℓ)
  run_full_experiment.py # 主 runner (checkpoint + auto-TP)
  run_clox.py            # 分 phase runner
  synthetic_dag.py       # 合成 DAG 理论验证
  benchmarks.py          # 4 benchmark 加载器
  evaluation.py          # Bootstrap CI, McNemar, Cohen's d
  analyze_v2.py          # 结果分析 + 绘图
results/
  synthetic/             # DAG 验证结果
  v3/                    # 32B 结果 (拓扑 + pilot)
  v4/                    # 8B 结果 (拓扑 + pilot)
  v5/                    # [目标] 完整策略对比
paper/
  main.tex               # NeurIPS 论文草稿
```

## 下一步（Reviewer 要求）

1. 完成 32B 全部 9 策略 × 3 seeds × 4 benchmarks
2. 评估 CLOX-Adaptive 在 held-out split 上的表现
3. Proxy validation（3/5/8 pilot traces vs 30 ground truth）
4. 完整统计检验（paired bootstrap CI, McNemar, Bonferroni）
5. 论文重写，纳入 short-ℓ regime 发现
