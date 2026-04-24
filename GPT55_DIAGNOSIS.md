# 审查范围与总判断

我可以访问公开 GitHub 仓库并读取 README、paper、docs、code、results、raw JSON/Markdown；但当前执行环境无法直接 `git clone` 或运行仓库，因此本报告是**静态代码与可见日志审计**，不是本地复现实验报告。凡是涉及“已验证”的地方，含义是“文件/日志可见且内部一致”，不是“我已重新运行得到相同结果”。

核心结论：当前项目不应继续把 **topology → fixed strategy routing** 或 **targeted repair pilot positive** 当主线。所有现象更一致地指向一个缺失机制：

> 当前仓库缺少的是 **calibrated, outcome-aware, per-instance portfolio selection with compute gating**：策略多样性确实偶尔包含正确答案，但现有方法没有可靠机制判断“哪个候选答案值得选、何时继续花 compute、何时退回 SC”。

唯一推荐的新主线是：

> **CLOX-PCS: Calibrated Portfolio Compute Selection**
> 用多策略作为候选生成器，而不是把某个旧策略当主方法；用严格 held-out calibration 学习候选质量、策略互补性、答案簇支持、成本和不确定性之间的关系；再用 value-of-compute gate 决定是否继续采样/验证，最终在统一预算下选择答案。

这不是“选择当前最好分支”。它继承的是**现象背后的机制线索**：oracle gap / 低相关策略对 / BAV efficiency signal 说明候选集合里有可利用信息；但 cross-strategy vote、topology rule、masked repair 都没有 selector/calibration，所以失败。

---

# 0. 仓库可读性判断

README 明确把项目描述为“CLOX v2 — 拓扑感知的计算最优推理策略选择”，核心 claim 是从少量 pilot traces 估算局部可恢复性 `r̄` 和误差传播长度 `ℓ`，自动路由到最优策略，并在匹配 Self-Consistency 准确率时节省 40–60% 计算开销。README 同时显示 full 9-strategy comparison、CLOX-Adaptive eval、proxy validation 仍是 pending。([GitHub][1])

| Item                     |  Found? | Location                                                               | Notes                                                                                    |
| ------------------------ | ------: | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Repository accessible    |     Yes | GitHub public repo                                                     | 可浏览代码与 raw 文件；不能本地 clone/run。                                                            |
| README                   |     Yes | `README.md`                                                            | 有项目 claim、进度、命令、结果索引。                                                                    |
| Paper draft              |     Yes | `paper/main.tex`, `paper/paper_draft.md`                               | `main.tex` 含强 claim：topology-dependent theory、CLOX-Adaptive 接近 oracle-best。([GitHub][2]) |
| Training scripts         | Partial | `code/run_full_experiment.py`, `code/run_clox.py`, `code/run_pilot.py` | 这是 inference-time experiment 仓库，不是训练模型主仓。                                                |
| Evaluation scripts       |     Yes | `code/evaluation.py`, `code/analyze_*.py`                              | metric 与 stats 有实现，但 answer checking 有污染风险。                                              |
| Configs                  | Partial | shell scripts + Python args + docs YAML                                | 未看到统一 Hydra/YAML config 主系统。                                                             |
| Logs/results             |     Yes | `results/`                                                             | 有 topology、pilot、BAV、meta、pdsc 等结果。                                                      |
| Baselines                |     Yes | CoT, SC, compute-matched SC, full regen, random repair                 | baseline 存在，但公平性/seed/metric 需修。                                                         |
| Failed experiments       |     Yes | `EXPERIMENTS.md`, `RESEARCH_BRIEF.md`, `IDEA_REPORT.md`                | 明确记录 ablation bug、pilot-to-scale collapse、topology thesis failure。([GitHub][3])          |
| Ablation                 |     Yes | `EXPERIMENTS.md`, `results/pilot`, strategy controls                   | 早期 ablation 被 code-path bug 污染；后续 ablation 多为 pilot。                                     |
| Requirements/environment |     Yes | `requirements.txt`, `setup.sh`, `code/requirements.txt`                | 可见但未执行安装。                                                                                |
| Checkpoints              | Partial | `.ckpt_*.json` mechanism in code; not persistent in final results      | Aggregator 会删除 checkpoint，影响审计。                                                          |
| Dataset manifests        |      No | not explicit                                                           | 缺少固定 sample IDs / split manifests。                                                       |
| WandB/TensorBoard        |      No | none visible                                                           | 似乎未用。                                                                                    |

| Missing Item                         | Why Needed                                                        | What I Should Upload                                         |
| ------------------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------------ |
| Full zip of repo at exact commit     | 便于本地 grep、line-accurate audit、test execution                      | 完整 zip 或 commit hash + archive                               |
| Full raw experiment logs             | 判断 command/config/seed/checkpoint 是否对应                            | `results/**/*.log`, stdout/stderr, scheduler logs            |
| Exact command history                | 防止 stale checkpoint / mixed env                                   | shell history or run manifest                                |
| Full per-example outputs             | 做 paired bootstrap、McNemar、error correlation、selector calibration | 不要只给 aggregate JSON；要每个 example × strategy × seed prediction |
| Dataset split manifest               | 防止 first-N bias 与 train/test leakage                              | 每个 benchmark 的 example IDs、source repo、split、shuffle seed    |
| Checkpoints before deletion          | 审计 stale checkpoint 和 resume 行为                                   | `.ckpt_*.json` 或 JSONL per-run                               |
| Paper compiled PDF / latest Overleaf | `main.tex` 可能不是最新版                                                | 最新 PDF + source                                              |
| Official baseline reproduction logs  | SOTA/fairness 必需                                                  | SC/BoN/Self-Certainty/FOBAR/RTR 等复现日志                        |

---

# 1. Repository Map

| Component                | Path                                                                        | Purpose                                              |   Importance | Notes                                                                                           |
| ------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------------- | -----------: | ----------------------------------------------------------------------------------------------- |
| README/project claim     | `README.md`                                                                 | 当前论文叙事、命令、进度                                         |         High | claim 强，但 full comparison pending。                                                              |
| Main paper               | `paper/main.tex`                                                            | NeurIPS-style manuscript                             |         High | 抽象和引言声称 real experiments support theory，但仓库结果未充分支持。                                             |
| Early experiment record  | `EXPERIMENTS.md`                                                            | 早期 synthetic/proxy 结果与失败 ablation                    |         High | 明确承认 two ablations identical，probable code-path bug。([GitHub][3])                               |
| Research brief           | `RESEARCH_BRIEF.md`                                                         | v2 lessons and v3 direction                          |         High | 直接记录 topology-to-strategy failure、pilot-to-scale collapse。([GitHub][4])                         |
| Idea report              | `IDEA_REPORT.md`                                                            | 新方向候选                                                |         High | 推荐 cross-strategy verification，但 pilot 后应修正为 selector/calibration，不是 naive voting。([GitHub][5]) |
| Full experiment runner   | `code/run_full_experiment.py`                                               | pilot/topology/strategies/proxy phases               |     Critical | seed、checkpoint、aggregation 影响所有结论。                                                             |
| Engine                   | `code/engine.py`                                                            | vLLM wrapper, answer extraction, step split          |     Critical | `extract_answer` 影响所有 metrics；engine seed default 42。                                           |
| Evaluation               | `code/evaluation.py`                                                        | answer checking, bootstrap, McNemar, efficiency      |     Critical | fallback `pred in ref or ref in pred` 可污染 metrics。([GitHub][6])                                 |
| Strategy implementations | `code/strategies_v2.py`                                                     | CoT, SC, targeted repair, random, backward, adaptive |     Critical | targeted/random repair prompt 实际更像 full rewrite；adaptive maps `adaptive` → targeted repair。     |
| Topology estimation      | `code/topology_v2.py`                                                       | `r̄`, `ℓ`, strategy recommendation                   |     Critical | recoverability proxy 是 agreement/confidence + optional mask-regeneration，不是严格 causal topology。  |
| Benchmark loaders        | `code/benchmarks.py`                                                        | GSM8K, MATH, StrategyQA, ARC, BBH                    |         High | StrategyQA 可 fallback 到 train；first-N sampling。([GitHub][7])                                    |
| Analysis scripts         | `code/analyze_*.py`                                                         | pilot/BAV/meta analysis                              |       Medium | 可保留，但需统一 result schema。                                                                         |
| Legacy code              | `code/main.py`, `methods.py`, `strategies.py`, `topology.py`, stage scripts | Medium/Low                                           | 多条旧路线，易混淆主线。 |                                                                                                 |
| Results v3/v4            | `results/v3`, `results/v4`                                                  | topology + pilot results                             |         High | v3/v4 topology 互相不稳定；v3 pilot positive 不应锚定。                                                    |
| BAV results              | `results/bav/bav_analysis.json`                                             | backward agreement verification                      |         High | BAV Pareto 有 efficiency signal，但不 beat SC。([GitHub][8])                                         |
| Meta sweep               | `results/meta/meta_summary.json`                                            | oracle-SC gap by model/task                          |         High | 证明 candidate pool has oracle room，但不是方法效果。([GitHub][9])                                         |
| Docs/stage notes         | `docs/`, `review-stage/`, `.aris/`                                          | research state / auto-review                         |       Medium | 有历史价值，但主线应冻结。                                                                                   |
| Tests                    | `tests/`, `verification/`                                                   | sanity/verification                                  |       Medium | 需要新增 metric/seed/leakage tests。                                                                 |

## 1.1 仓库当前试图解决的问题

给定 LLM reasoning problem 和固定 inference-time compute budget，选择最合适的推理策略：single CoT、SC、masked repair、backward cloze、full regeneration、hierarchical/adaptive 等。README 和 paper 将这个问题解释为由 `r̄` 和 `ℓ` 决定的 topology-aware strategy selection。([GitHub][1])

## 1.2 当前已有方法

当前 paper 主方法是 **CLOX-Adaptive**：先生成 pilot CoT traces，估计 `r̄` / `ℓ`，再根据阈值选择策略。代码中 `CLOXAdaptive` 运行 topology estimation，然后执行选中策略；若 topology 返回 `"adaptive"`，代码直接映射到 `"targeted_repair"`。([GitHub][10])

## 1.3 当前方法核心假设

1. `r̄` 和 `ℓ` 可以从少量 traces 可靠估计。
2. 它们能预测哪类 inference strategy 最优。
3. targeted masked repair 在高 recoverability / 短 propagation regimes 优于 SC。
4. SC 在低 recoverability / 长 propagation regimes 更优。

这些假设在 paper 中以 theorem/selector 形式出现；但 `RESEARCH_BRIEF.md` 明确记录现实中所有 benchmark `ℓ` 都短、SC-dominant regime 未出现、pilot-to-scale collapse，并且 SC-5 在 GSM8K scale 上 dominates。([GitHub][2])

## 1.4 主线文件、历史遗留与会影响结论的文件

主线文件：`code/run_full_experiment.py`, `code/strategies_v2.py`, `code/topology_v2.py`, `code/evaluation.py`, `code/benchmarks.py`, `code/engine.py`, `results/v3`, `results/v4`, `results/pilot`, `results/bav`, `results/meta`, `paper/main.tex`.

历史遗留/混淆文件：`code/README.md`, `code/main.py`, `code/methods.py`, `code/strategies.py`, `code/topology.py`, `stage-13*`, `run_quick_7b.py`, `run_focused.py`, many docs/stage JSONs.

直接影响实验结论的代码：answer extraction/checking、benchmark split、seed setting、checkpoint resume/delete、strategy prompts、token accounting、aggregation/statistics。

---

# 2. Result Reliability Audit

状态含义：这里的 Verified/Partially Verified 指“可见文件存在并可解析”，不是我本地重跑。

| Result ID | Result Name                         | Dataset                              | Metric                            |                                               Claimed Value |                                                          Logged Value | Config  | Seed                   | Command                    | Checkpoint                           | Status                     | Reliability                                      | Issue                                                                                                                         |
| --------- | ----------------------------------- | ------------------------------------ | --------------------------------- | ----------------------------------------------------------: | --------------------------------------------------------------------: | ------- | ---------------------- | -------------------------- | ------------------------------------ | -------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| R0        | README synthetic DAG                | synthetic DAG                        | theory prediction accuracy        |                                                         83% |                                                           README only | partial | 3 seeds claimed        | `python3 synthetic_dag.py` | unknown                              | Partially Verified         | medium                                           | 未审计 raw trial table。                                                                                                          |
| R1        | Early CLOX proxy                    | 64 examples mixed                    | accuracy                          | SC best 77.60%; targeted=random 76.04%; backward=CoT 75.52% |                                                      `EXPERIMENTS.md` | partial | 5 seeds claimed        | missing exact command      | none                                 | Possibly Contaminated      | unusable for positive claims                     | 文档承认 ablations identical across 9 metrics，probable code-path bug。([GitHub][3])                                                |
| R2        | v3 topology                         | GSM8K/MATH/StrategyQA/ARC ×200       | `r̄`, `ℓ`, predicted distribution |                                             topology varies |                                                          JSON visible | partial | pilot samples internal | README command approximate | no raw traces                        | Partially Verified         | medium for proxy values, low for strategy claims | Model path mismatch: README says `Qwen3.5-27B`, result path is `Qwen2.5-32B-Instruct-AWQ`; proxy not validated.([GitHub][11]) |
| R3        | v4 topology                         | MATH/StrategyQA/ARC ×200             | `r̄`, `ℓ`                         |                                             topology varies |                                                          JSON visible | partial | unknown                | unknown                    | no raw traces                        | Partially Verified         | medium/low                                       | StrategyQA `r̄` flips high vs v3; model/task sensitivity not explained.([GitHub][12])                                         |
| R4        | v3 pilot                            | 4 benchmarks ×50                     | accuracy/tokens                   |                              targeted positive on GSM8K/ARC |                                GSM8K targeted 0.98; ARC targeted 0.90 | partial | unclear                | missing exact command      | summary only                         | Partially Verified         | low/medium                                       | n=50, first-N risk, later research brief says pilot-to-scale collapse.([GitHub][13])                                          |
| R5        | Pilot 8-strategy analysis           | 50 problems                          | accuracy, phi, voting             |                           proceed due low-correlation pairs | SC 0.88, oracle_any 0.90, cross_K5 best 0.82, topology correlation ~0 | partial | unclear                | missing                    | per-example likely visible elsewhere | Partially Verified         | medium as diagnostic, low as performance claim   | Shows selector gap, not main method success.([GitHub][14])                                                                    |
| R6        | BAV analysis                        | 50 problems                          | acc/tokens                        |                                       BAV Pareto efficiency |     BAV 0.80 @7522 tokens; SC 0.88 @17360; agreed/disagreed both 0.80 | partial | unknown                | missing                    | summary only                         | Partially Verified         | medium diagnostic                                | BAV gate has no discriminative power; not accuracy winner.([GitHub][8])                                                       |
| R7        | Meta oracle gap                     | models × tasks ×30                   | oracle-SC gap                     |                                                     proceed |                                      max gap 0.333; 11 cells gap≥0.15 | partial | unknown                | scripts exist              | per-cell files partial               | Partially Verified         | medium as opportunity signal                     | Oracle uses labels; cannot be reported as method.([GitHub][9])                                                                |
| R8        | Full 9-strategy comparison          | 4 benchmarks ×200 ×3 seeds           | accuracy/tokens/statistics        |                                                     pending |                                                    not run per README | N/A     | planned                | command in README          | N/A                                  | Missing Log                | unusable                                         | This is the critical experiment; not available.([GitHub][1])                                                                  |
| R9        | CLOX-Adaptive eval                  | held-out                             | accuracy/token saving             |                                                     pending |                                                    not run per README | N/A     | planned                | unknown                    | N/A                                  | Missing Log                | unusable                                         | Paper claim unsupported.                                                                                                      |
| R10       | Proxy validation                    | topology proxy vs ground truth       | Spearman/Pearson                  |                                                     pending |                                                    not run per README | N/A     | planned                | command in README          | N/A                                  | Missing Log                | unusable                                         | Core topology estimator not validated.([GitHub][1])                                                                           |
| R11       | Paper abstract real benchmark claim | 5 benchmarks, 3 models, 9 conditions | near-oracle                       |                            ≤15% overhead, oracle-best match |                                       no corresponding result visible | missing | missing                | missing                    | missing                              | Contradicted / Missing Log | unusable                                         | Paper overclaims relative to README progress.([GitHub][2])                                                                    |
| R12       | StrategyQA/ARC meta rows            | ARC etc                              | answer correctness                |                                                     various |                  some predictions look non-letter/textual yet correct | partial | unknown                | missing                    | partial                              | Possibly Contaminated      | low                                              | Multiple-choice / extraction may inflate metrics.                                                                             |

**Reliability conclusion:**
Strong evidence supports “old topology-to-strategy thesis is not currently proven and likely failed at scale.” Medium evidence supports “strategy diversity creates oracle room.” Low evidence supports any specific existing method as a winner.

---

# 3. Code Correctness Audit

| Priority | File                          | Function/Class                              | Code Region                      | Suspicion                                                                                                                                                                | Evidence                                                                                                                                                                                                       | How to Verify                                                                                                                      | Proposed Fix for Claude Code                                                                                                                                                                       | Expected Effect                                                   | Confidence  |
| -------: | ----------------------------- | ------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------- |
|       P0 | `code/evaluation.py`          | `check_answer`                              | multiple-choice + final fallback | Metric contamination: if MC regex fails, function falls through to `return pred in ref or ref in pred`; for text/MC this can mark partial junk correct.                  | `check_answer` normalizes, handles MC regex, then returns substring containment for all remaining cases.([GitHub][6]) Meta result rows include ARC predictions that appear non-letter/text yet marked correct. | Add unit tests: prediction `"1"` vs ref `"D"` false; `"the main reason is..."` vs `"C"` false unless exact extracted option label. | Split `check_answer_by_type`; MC must compare extracted label only; text must exact/normalized or benchmark-specific official metric; remove generic substring fallback except explicitly allowed. | May lower ARC/BBH/StrategyQA numbers; increases trust.            | high        |
|       P0 | `code/run_full_experiment.py` | `run_strategies`, `init_engine`             | seed loop                        | Seeds are metadata, not actual deterministic controls. Engine initialized once with default seed; seed not passed to strategy/engine; no `random/np/torch` seed setting. | `VLLMEngine` has seed arg default 42; `init_engine` does not pass per-run seed. Seed is only recorded in result dict.([GitHub][15])                                                                            | Run two seeds on same 5 examples with deterministic temp and compare output; inspect vLLM sampling params.                         | Add `set_global_seed(seed)`, re-init or parameterize vLLM seed per run where possible; pass `seed` into stochastic strategies; log seed and sampling params.                                       | True multi-seed variance; possible change in all stds.            | high        |
|       P0 | `code/run_full_experiment.py` | `_aggregate_benchmark`                      | checkpoint cleanup               | Deletes `.ckpt_*.json` raw per-example results after aggregation; aggregate JSON lacks raw predictions, making paired tests and audit impossible.                        | Code writes aggregate `strategies.json`, then unlinks `.ckpt_*.json`.([GitHub][16])                                                                                                                            | Run tiny strategy phase and inspect output directory.                                                                              | Preserve immutable JSONL: `per_example/{benchmark}/{strategy}_s{seed}.jsonl`; never delete raw unless archived.                                                                                    | Enables stats, calibration, debugging.                            | high        |
|       P0 | `code/benchmarks.py`          | `load_strategyqa`                           | dataset source selection         | StrategyQA silently tries `ChilleD/StrategyQA` test then `metaeval/strategy-qa` train; evaluation could use train split without explicit flag.                           | Loader loops over `(repo, split)` including `("metaeval/strategy-qa","train")`.([GitHub][7])                                                                                                                   | Log `repo`, `split`, dataset fingerprint in every run manifest.                                                                    | Require explicit `--allow_train_eval false`; fail if test unavailable unless `--dev_mode`. Save split manifest.                                                                                    | Prevent leakage / incomparable numbers.                           | high        |
|       P0 | `code/strategies_v2.py`       | `UncertaintyTargetedRepair`, `RandomRepair` | repair prompt                    | “Selective repair” is actually complete solution rewrite; targeted vs random cannot isolate local repair mechanism.                                                      | Targeted prompt says “Rewrite the complete solution”; random says “Complete the solution”; both can rewrite all downstream reasoning.([GitHub][10])                                                            | Log edit distance, number of unmasked steps changed; compare targeted/random/full regen outputs.                                   | Implement strict masked-slot filling with JSON output for masked indices only; rebuild solution; log changed steps. Or archive as full-rewrite ablation.                                           | Likely removes false targeted-repair signal; clarifies mechanism. | high        |
|       P1 | `code/strategies_v2.py`       | `RandomRepair`                              | RNG                              | Uses Python `hash(question)`; Python hash is process-randomized unless `PYTHONHASHSEED` fixed; seed loop ignored.                                                        | Code uses `np.random.default_rng(hash(question) % ... + 12345)`.([GitHub][10])                                                                                                                                 | Run same command in two processes and compare masked indices.                                                                      | Use stable hash `hashlib.sha256(question + seed)`; pass seed.                                                                                                                                      | Reproducibility.                                                  | high        |
|       P1 | `code/topology_v2.py`         | `estimate_recoverability`                   | `r̄` estimator                   | `r̄` is lexical step agreement + confidence, not true local recoverability; optional mask test gives future/downstream steps, leaking context.                           | Code combines `0.6*agreement + 0.4*confidence`; mask-regenerate prompt includes all non-masked steps, including downstream/final context.([GitHub][17])                                                        | Create synthetic traces where lexical agreement high but answer wrong; inspect r_bar.                                              | Rename to `topology_proxy_features`; add causal ablation excluding downstream and final answer; validate proxy vs actual strategy wins.                                                            | Removes unjustified theoretical claim.                            | high        |
|       P1 | `code/strategies_v2.py`       | `CLOXAdaptive`                              | selected strategy                | If topology returns `"adaptive"`, code maps it to `"targeted_repair"`; topology summary mostly returns adaptive, so actual selector collapses to targeted repair.        | `selected = topo.strategy; if selected == "adaptive": selected = "targeted_repair"`.([GitHub][10]) v3 topology distribution is mostly `"adaptive"`.([GitHub][11])                                              | Log selected concrete strategy on 100 examples.                                                                                    | Do not allow `"adaptive"` as a strategy label; produce concrete actions only. Under new path, freeze CLOXAdaptive as legacy baseline.                                                              | Prevents misleading adaptive claims.                              | high        |
|       P1 | `code/run_full_experiment.py` | `run_pilot`, `run_topology`                 | example selection                | Uses first `max_examples`; pilot-to-scale collapse suggests first-N easy bias.                                                                                           | `load_benchmark(... max_examples=n)` iterates from start; research brief says targeted_repair 98% n=50 → 70.5% n=200.([GitHub][16])                                                                            | Compare first-50 vs seeded random-50 difficulty/accuracy.                                                                          | Add fixed ID manifest and seeded stratified sampling.                                                                                                                                              | More honest pilot.                                                | high        |
|       P1 | `code/engine.py`              | `extract_answer`                            | regex/fallback                   | Extraction may return arbitrary short line or first number, affecting MC/text tasks.                                                                                     | `extract_answer` falls back to last short line/number; no answer-type awareness.([GitHub][18])                                                                                                                 | Unit tests by benchmark type.                                                                                                      | Move extraction to `extract_answer(text, answer_type, choices=None)`; log raw + extracted.                                                                                                         | Cleaner metrics.                                                  | medium/high |
|       P2 | `code/run_full_experiment.py` | token accounting                            | `CLOXAdaptive`                   | Pilot token cost estimated as `n_pilot * max_tokens`, not actual generated tokens.                                                                                       | Code comment: “upper bound estimate”.([GitHub][10])                                                                                                                                                            | Compare actual pilot outputs vs reported tokens.                                                                                   | Return actual pilot token counts from topology estimation.                                                                                                                                         | Fair token comparisons.                                           | medium      |
|       P2 | `docs/novelty_report.json`    | novelty pipeline                            | docs                             | Claims no similar papers may be stale/overoptimistic; external search finds many close works.                                                                            | IDEA_REPORT itself says entropy/topology ideas crowded/scooped.([GitHub][5])                                                                                                                                   | Manual related-work audit.                                                                                                         | Archive automated novelty report as historical.                                                                                                                                                    | Avoid novelty overclaim.                                          | high        |

---

# 4. Claim–Code–Result Matrix

| Claim                                                 | Source File                | Implementation File              | Result Evidence                                         | Status                         | Problem                                                                                      | Confidence |
| ----------------------------------------------------- | -------------------------- | -------------------------------- | ------------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------- | ---------- |
| `r̄` and `ℓ` determine optimal strategy               | README, paper              | `topology_v2.py`, `CLOXAdaptive` | v3/v4 topology summaries; pilot topology correlation ~0 | Contradicted / Unclear         | Real data show short `ℓ` everywhere and topology not predictive in pilot; proxy unvalidated. | high       |
| CLOX saves 40–60% compute while matching SC           | README                     | `CLOXAdaptive`, strategy runner  | no full adaptive eval                                   | Unsupported                    | README progress says adaptive eval pending.                                                  | high       |
| Masked repair beats SC in high recoverability regimes | paper theorem + method     | `UncertaintyTargetedRepair`      | early ablations contaminated; pilot positive unstable   | Unsupported                    | Repair prompt not local; targeted=random in early result.                                    | high       |
| No fixed strategy dominates                           | paper                      | strategy suite                   | some pilot/meta evidence                                | Partially Supported            | SC dominates several visible settings; full comparison missing.                              | medium     |
| Topology estimates can be computed from pilot traces  | paper/code                 | `estimate_topology`              | topology JSON visible                                   | Supported as proxy computation | Values computed, but causal meaning not established.                                         | high       |
| Topology predicts winning strategy                    | paper                      | threshold selector               | pilot topology correlation near zero; scale failure     | Contradicted                   | `r_bar_vs_disagreement` ≈ 0 in pilot analysis.                                               | high       |
| BAV/cross-strategy diversity improves compute         | `IDEA_REPORT`, BAV scripts | `analyze_bav.py` / results       | BAV Pareto efficiency but not SC accuracy               | Partially Supported            | BAV 0.80 vs SC 0.88; gate not discriminative.                                                | medium     |
| Cross-strategy independence exists                    | `IDEA_REPORT`              | pilot analysis                   | 6/28 low-correlation pairs; mean abs phi 0.449          | Partially Supported            | Diversity exists but too weak for naive vote to beat SC.                                     | medium     |
| Oracle gap proves opportunity                         | `results/meta`             | `meta_sweep.py`                  | max oracle-SC gap 0.333; 11 cells ≥0.15                 | Supported as opportunity only  | Oracle is label-using, not deployable method.                                                | high       |
| Existing full benchmark supports paper                | paper abstract             | full runner                      | README says pending                                     | Unsupported                    | Strong paper claims must be deleted/replaced.                                                | high       |

---

# 5. Phenomenon Ledger

| ID  | Observation                                                                               | Type                  | Where Found                         | Setting               | Metric                | Compared To                      | Reliability    | What It Suggests                                                          | What It Rules Out                           | Confidence  |
| --- | ----------------------------------------------------------------------------------------- | --------------------- | ----------------------------------- | --------------------- | --------------------- | -------------------------------- | -------------- | ------------------------------------------------------------------------- | ------------------------------------------- | ----------- |
| P01 | Budget-matched SC best in early 64-example result; targeted=random; backward=CoT          | Negative/Mixed        | `EXPERIMENTS.md`                    | synthetic/proxy mixed | accuracy              | CoT/repair                       | low due bug    | Extra samples help more than current masking                              | Current masking mechanism as implemented    | high        |
| P02 | Two key ablations identical across all 9 metrics                                          | Anomalous             | `EXPERIMENTS.md`                    | early ablations       | all metrics           | forward/backward, targeted/whole | high as report | Condition-specific code likely not triggered                              | Any causal claim from those ablations       | high        |
| P03 | Low-confidence examples have high error for all methods                                   | Negative              | `EXPERIMENTS.md`                    | early                 | error split           | all methods                      | medium/low     | Need confidence/calibration, not just repair                              | Raw entropy targeting alone                 | medium      |
| P04 | v3 pilot targeted_repair strong on GSM8K/ARC                                              | Positive but unstable | `results/v3/.../pilot`              | n=50                  | accuracy              | SC/CoT                           | low/medium     | Repair/full rewrite may sometimes fix answers                             | General targeted repair claim               | medium      |
| P05 | Pilot-to-scale collapse: targeted 98% n=50 → 70.5% n=200                                  | Negative              | `RESEARCH_BRIEF.md`                 | GSM8K                 | accuracy              | SC-5                             | medium         | First-N/easy subset and overfit likely                                    | Pilot positive as final method              | high        |
| P06 | All real benchmarks have short `ℓ`; SC-dominant regime absent                             | Negative              | `RESEARCH_BRIEF.md`, topology JSON  | v2/v3                 | `ℓ`                   | theory regimes                   | medium         | Theory partition mismatched to real traces                                | Hard threshold topology router              | high        |
| P07 | v3/v4 topology values differ sharply, e.g. StrategyQA `r̄` low in v3, high in v4          | Unstable              | topology summaries                  | model variants        | `r̄`                  | v3 vs v4                         | medium         | Proxy model-sensitive / unstable                                          | Task-intrinsic topology claim               | high        |
| P08 | CLOXAdaptive mostly receives `"adaptive"` recommendation; code maps it to targeted repair | Anomalous             | topology JSON + code                | v3/v4                 | strategy_distribution | concrete strategy                | high           | Selector degenerates                                                      | Claim of principled routing                 | high        |
| P09 | Pilot analysis: oracle_any 0.90, SC 0.88, cross_K5 0.82                                   | Mixed                 | `results/pilot/pilot_analysis.json` | 50 problems           | accuracy/tokens       | SC                               | medium         | Candidate pool has some complementary correctness, but naive voting fails | Cross-strategy voting alone                 | high        |
| P10 | Pairwise error correlation: 6/28 low, mean abs phi 0.449                                  | Mixed                 | pilot analysis                      | 8 strategies          | phi                   | independence threshold           | medium         | Some diversity exists, not enough independence                            | Assumption of broad independence            | medium/high |
| P11 | Topology correlations with disagreement/diversity near zero                               | Negative              | pilot analysis                      | 50 problems           | Pearson               | disagreement/diversity           | medium         | `r̄/ℓ` not useful as sole predictor                                       | Topology-driven selection as main mechanism | high        |
| P12 | BAV 0.80 @ 7522 tokens, SC 0.88 @ 17360; acc_agreed=acc_disagreed=0.80                    | Mixed                 | `results/bav`                       | 50 problems           | acc/tokens            | SC                               | medium         | There is efficiency signal, but agreement gate is uncalibrated            | BAV as final method                         | high        |
| P13 | Meta oracle gap max 0.333, 11 cells ≥0.15                                                 | Positive signal       | `results/meta`                      | models/tasks          | oracle-SC gap         | SC                               | medium         | There is answer-level selection opportunity                               | Reporting oracle as method                  | high        |
| P14 | StrategyQA chance / ARC saturation at scale                                               | Negative              | `RESEARCH_BRIEF.md`                 | scale                 | accuracy              | strategies                       | medium         | Benchmarks can be unsuitable for demonstrating mechanism                  | Cherry-picking only GSM8K/MATH              | high        |
| P15 | Multiple-choice answer extraction likely permissive                                       | Anomalous             | code + meta rows                    | ARC                   | correctness           | labels                           | medium/high    | Need metric refactor before method claims                                 | Any ARC result as high-confidence now       | high        |

---

# 6. Design Constraints

| Constraint ID | Derived From Observation | Constraint Type    | Meaning                                                              | Implication for New Method                                                                                                   | Confidence  |
| ------------- | ------------------------ | ------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------- |
| C01           | P01/P02                  | Must Avoid         | Do not claim current masked repair mechanism works.                  | Keep repair only as candidate generator/ablation until strict masked-slot implementation exists.                             | high        |
| C02           | P04/P05                  | Must Stabilize     | Pilot positives are fragile.                                         | Use held-out calibration/test split and seeded random IDs; never first-N-only.                                               | high        |
| C03           | P06/P11                  | Must Not Claim     | `r̄/ℓ` do not currently predict winners.                             | Reframe topology as auxiliary features, not theorem-backed router.                                                           | high        |
| C04           | P09/P13                  | Must Preserve      | Candidate pool sometimes contains correct answers.                   | New method should exploit oracle gap via selection, not voting.                                                              | high        |
| C05           | P10/P12                  | Must Fix           | Diversity exists but current gate/vote cannot select.                | Add calibrated selector and value-of-compute gate.                                                                           | high        |
| C06           | P12                      | Must Control       | Compute efficiency matters; high accuracy via huge SC is not enough. | Optimize accuracy–token Pareto, not accuracy only.                                                                           | high        |
| C07           | P15                      | Must Fix           | Metrics may be contaminated.                                         | Metric/unit-test refactor before any new result.                                                                             | high        |
| C08           | P03                      | Must Calibrate     | Uncertainty is useful only if calibrated to correctness.             | Add Brier/ECE and answer-cluster calibration logs.                                                                           | medium/high |
| C09           | P14                      | Must Generalize    | Avoid dataset-specific “success”.                                    | Include math, MC, boolean, BBH/logical or held-out tasks.                                                                    | high        |
| C10           | R8/R9/R10                | Must Test          | Full comparison/adaptive/proxy are missing.                          | Minimal experiments must first reproduce best positive/negative fragments, then test new mechanism.                          | high        |
| C11           | Code audit               | Must Preserve      | Raw per-example results are essential.                               | New result schema must save every candidate, feature, score, cost, decision.                                                 | high        |
| C12           | Related work             | Must Differentiate | Adaptive routing/BoN/self-certainty already exist.                   | Novelty must be “cross-strategy calibrated portfolio + value-of-compute + correlation-aware selection,” not generic routing. | high        |

---

# 7. Negative-to-Insight Analysis

| Negative Observation                        | Failed Assumption                                  | Why the Assumption Failed                                                                     | What Mechanism Is Missing                               | New Design Requirement                                                                    |
| ------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Targeted repair ties random/whole-mask      | Entropy-targeted local repair is active ingredient | Prompt permits full rewrite; entropy may not identify causal error; metric/ablation bug       | Causal edit constraint + outcome-aware validation       | Treat repair as candidate source; verify wrong→right transitions and strict edit locality |
| Backward cloze no gain / ablation identical | Backward reconstruction adds useful verification   | Backward prompt not selecting among candidates; compare may be string-level; compute not fair | Candidate verifier/reranker calibrated to correctness   | Use backward verification as feature, not final answer generator                          |
| Pilot-to-scale collapse                     | n=50 pilot reflects benchmark                      | First-N/easy bias, no held-out split, no stable seed                                          | Split-aware calibration and test protocol               | Fixed sample manifests; dev/calibration/test separation                                   |
| Topology not predictive                     | `r̄/ℓ` are sufficient strategy statistics          | Proxy measures lexical agreement/confidence, not actual correctness; real `ℓ` compressed      | Empirical selector that learns when features matter     | Use topology features only if they improve held-out selector                              |
| Cross-strategy voting underperforms SC      | Error diversity alone beats SC                     | Errors still correlated; voting ignores candidate quality                                     | Calibrated answer/candidate scorer                      | Learn scores from features and answer clusters                                            |
| BAV gate not discriminative                 | Agreement/disagreement identifies reliability      | agreed and disagreed subsets same accuracy                                                    | Calibration and value-of-information estimation         | Gate should predict expected gain from more compute                                       |
| StrategyQA chance / ARC saturation          | Any benchmark will show mechanism                  | Some tasks too noisy/saturated; metrics suspect                                               | Task-aware evaluation and benchmark gating              | Pre-register tasks and success/failure interpretation                                     |
| SC dominates scale                          | Repair is core compute-efficient alternative       | Current repair not reliable; SC majority is strong baseline                                   | Selectively use portfolios only when expected gain > SC | Always compare to compute-matched and full SC                                             |

---

# 8. Method Synthesis Table

| Evidence Fragment                | Source in Repo                               | What It Reveals                                                  | Generalized Principle                              | Use in New Method?          | How to Transform It                                                  |
| -------------------------------- | -------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------- | --------------------------- | -------------------------------------------------------------------- |
| `r̄/ℓ` topology summaries        | `results/v3`, `results/v4`, `topology_v2.py` | Structural proxies are measurable but unstable/predictively weak | Some trace-level features may help calibration     | Yes, auxiliary only         | Rename to `topology_proxy_features`; log but do not hard-route       |
| Targeted repair pilot wins       | v3 pilot                                     | Rewrite/repair sometimes produces alternative correct answer     | Candidate diversity can help                       | Yes, as generator/ablation  | Constrain or label as rewrite candidate; never claim local repair    |
| Random repair ties targeted      | early + pilot                                | Mask location may not be active ingredient                       | Diversity > targeting                              | Yes, negative control       | Keep as ablation to show selector uses quality not mask label        |
| Backward cloze moderate in pilot | pilot/BAV                                    | Backward-style reasoning can yield complementary candidate       | Verification direction may add features            | Yes, transformed            | Use backward verifier score, not answer alone                        |
| SC dominance                     | RESEARCH_BRIEF + pilot                       | Strong baseline and default fallback                             | Any method must beat or save compute vs SC         | Yes, baseline/gate fallback | PCS stops or falls back when selection confidence low                |
| Oracle gap                       | `results/meta`                               | Candidate set contains latent correct answers                    | Main bottleneck is selection, not generation alone | Yes, central                | Train/test calibrated selector to convert oracle room into real gain |
| Low-correlation pairs            | pilot_analysis                               | Diversity exists locally                                         | Portfolio should choose subset adaptively          | Yes                         | Correlation-aware portfolio features                                 |
| BAV Pareto but weak gate         | `results/bav`                                | Efficiency possible, gate crude                                  | Need calibrated gate                               | Yes, transformed            | Replace agreement gate with expected value-of-compute gate           |
| Metric bugs                      | `evaluation.py`, meta rows                   | Current results may inflate                                      | Evaluation must precede method                     | Yes, mandatory fix          | Add answer-type-specific extract/check tests                         |
| Full comparison missing          | README                                       | Core evidence absent                                             | Method must be validated minimally first           | Yes                         | Minimal queue before full benchmark                                  |

---

# 9. Missing Mechanism Diagnosis

1. **Missing Mechanism Name:**
   **Calibrated Outcome-Aware Portfolio Selection with Value-of-Compute Control**

2. **One-Sentence Diagnosis:**
   当前仓库已经能生成多种推理候选，但缺少一个可校准、可审计、不会偷看标签的机制来判断哪个候选答案更可能正确，以及何时继续花 compute。

3. **Evidence From Positive Results:**
   `results/meta` 显示多个 model/task cell 存在 oracle-SC gap，最大 0.333，11 个 cell ≥0.15；pilot 中 `oracle_any=0.90` 而 SC=0.88，说明候选集合偶尔包含 SC 没选中的正确答案。([GitHub][9])

4. **Evidence From Negative Results:**
   cross-strategy vote、BAV、targeted repair 都没有稳定 beat SC；BAV 的 agreed/disagreed accuracy 同为 0.80，说明简单 agreement gate 不足。([GitHub][8])

5. **Evidence From Unstable Results:**
   v3/v4 topology 不稳定，pilot-to-scale collapse，StrategyQA/ARC 出现 chance/saturation。([GitHub][11])

6. **Evidence From Failed Ablations:**
   早期 ablation identical，targeted=random，说明旧模块解释不可信。([GitHub][3])

7. **Why Existing Method Cannot Solve It:**
   CLOXAdaptive 是 topology-threshold router；它没有对候选答案进行 outcome-aware scoring，也没有学习何时继续采样，且 `"adaptive"` fallback 到 targeted repair。

8. **Why Simple Tuning Cannot Solve It:**
   调阈值不能解决 metric contamination、selector absence、pilot bias、correlated errors、candidate quality scoring 等结构问题。

9. **Why Existing Best Positive Fragment Is Insufficient:**
   targeted_repair pilot positive 可能来自 first-N/easy subset 或 full rewrite；BAV 有 Pareto signal 但不 beat SC；cross-vote更差。它们都没有校准 selector。

10. **What New Mechanism Must Do:**
    将多策略输出转成候选集合，记录每个候选的 answer cluster、strategy source、logprob/confidence、vote support、backward/verification signal、topology proxy、cost；在 held-out calibration split 上学习候选正确性；测试时用 value-of-compute gate 决定 stop/continue/select。

11. **Confidence:**
    **medium-high**. 对“旧 topology-repair 主线不足”的信心 high；对“PCS 是最可能成功新路径”的信心 medium，因为需要最小实验验证 selector 是否能把 oracle gap 转化为真实 gain。

---

# 10. New MAIN METHOD PATH

## Method Name Placeholder

**CLOX-PCS: Calibrated Portfolio Compute Selection**

## One-Sentence Core Idea

把 CoT/SC/repair/backward/full-regeneration 等旧策略降级为**候选生成器**，再用 held-out calibrated selector 和 value-of-compute gate 在每个 instance 上选择答案或追加计算。

## Core Missing Mechanism It Adds

**Outcome-aware calibrated selection + compute gating**：不是预测“哪种策略理论上最优”，而是预测“当前候选答案/答案簇是否可靠，以及多花 token 的期望收益是否超过成本”。

## What Phenomena It Explains

* Oracle gap 说明候选池有潜在收益，但 naive vote 失败说明缺 selector。
* BAV Pareto 说明少量 verification 可以提高效率，但 agreed/disagreed 同准确说明 gate 未校准。
* SC dominance 说明当 selector 不确定时应退回 SC，而不是强行 repair。
* topology 相关性弱说明 `r̄/ℓ` 只能作 features，不能作硬规则。
* targeted pilot collapse 说明 first-N 和不校准策略不能泛化。

## What Negative Results It Fixes

| Negative                | PCS Fix                                                |
| ----------------------- | ------------------------------------------------------ |
| targeted=random         | 不再把 target policy 当主贡献；仅作为 candidate source / ablation |
| topology not predictive | topology 变成 optional feature，经 held-out selector 验证    |
| cross-vote < SC         | 用 calibrated answer selection 替代 majority vote         |
| BAV gate weak           | 用 calibrated EVI gate 替代 agreement heuristic           |
| pilot collapse          | 严格 calibration/test split + fixed sample manifests     |
| metric contamination    | 先修 evaluation，再运行 PCS                                  |

## What Existing Positive Signals It Generalizes

不是“targeted_repair 有效”，而是：

> **不同策略有时产生互补候选；但互补性只有在被校准 selector 捕获时才有价值。**

## Why Existing Best Path Is Not Enough

现有 best positive fragment 可能是 targeted pilot、BAV Pareto 或 oracle gap。它们分别缺乏：稳定性、准确性、deployable selector。PCS 的主贡献是把这些碎片统一为一个可验证机制。

## Core Mechanism

1. **Scout portfolio:** 低成本生成多个不同策略候选。
2. **Feature ledger:** 为每个候选记录 answer cluster、strategy source、support、confidence、entropy、cost、topology proxy、backward/self-verifier score。
3. **Calibrated selector:** 在 calibration split 上学习 `P(candidate correct | features, history)`。
4. **Value-of-compute gate:** 若当前最佳答案簇置信度足够高则 stop；否则选择最可能提高正确率/成本比的下一策略。
5. **Final answer selection:** 选择得分最高的 answer cluster，而不是单条 trace 或多数票。

## New Objective / Loss

[
L_{\text{total}}
================

L_{\text{rank}}
+
\alpha L_{\text{cal}}
+
\beta L_{\text{cost}}
+
\gamma L_{\text{abstain}}
]

where:

[
L_{\text{rank}}
===============

-\log
\frac{\sum_{c \in C(x): z_c=1}\exp(s_\theta(c,H_x))}
{\sum_{c \in C(x)}\exp(s_\theta(c,H_x))}
]

[
L_{\text{cal}} = \text{Brier}(p_\theta(c), z_c) + \text{ECE}
]

[
L_{\text{cost}} = \max(0, \text{cost}(H_x)-B)
]

`z_c=1` means candidate answer is correct on calibration split only. At test time labels are unavailable.

## New Architecture or Module

* `PortfolioGenerator`
* `CandidateFeatureExtractor`
* `CalibratedSelector`
* `ValueOfComputeGate`
* `PortfolioRunLogger`
* `AnswerClusterer`

No LLM fine-tuning required for minimal version. Selector can start as logistic regression / isotonic calibration / small sklearn model over fixed features.

## New Training Procedure

1. Build fixed calibration/test manifests.
2. Run candidate portfolio on calibration set.
3. Fit selector on calibration candidates.
4. Freeze selector.
5. Evaluate on held-out test set under same budget.

## New Evaluation Protocol

Core comparison must include:

A. **Existing Best Positive Fragment Only** = current best deployable non-SC positive fragment, likely BAV/cross-strategy vote or targeted_repair depending benchmark.
B. **New MAIN METHOD Without New Mechanism** = same portfolio candidates, but majority vote / uncalibrated heuristic.
C. **Full New MAIN METHOD** = calibrated selector + value-of-compute gate.

PCS must also compare against Standard CoT, compute-matched SC, SC-k5/k8, full SC, BAV, targeted_repair, random_repair, backward_cloze.

## Existing Components It Reuses

`VLLMEngine`, benchmark loaders after split fixes, strategy implementations as generators, evaluation stats after metric fix, meta/pilot analysis code as diagnostic.

## Existing Components It Deletes / Rewrites / Archives

* Archive topology-threshold main claim.
* Rewrite evaluation/extraction.
* Rewrite result schema.
* Freeze CLOXAdaptive as legacy baseline.
* Keep targeted repair only as generator/ablation until strict repair exists.

## Why This Is Mechanism-Level Different from Prior Work

PCS is not just SC, not just BoN, not just self-certainty, and not generic model routing. It specifically studies **cross-strategy candidate portfolios** and learns when heterogeneous reasoning strategies add selection value beyond single-strategy sampling. However, novelty risk is real because adaptive routing and Best-of-N selection are crowded.

## Main Risk

The selector may not generalize: if candidates are too correlated or features cannot predict correctness, PCS will collapse to SC and should be reported as negative, not cherry-picked.

## Minimal Falsification Experiment

On a fixed 100-example held-out test after 50-example calibration:

* A: BAV/cross-vote
* B: same portfolio + majority vote
* C: PCS calibrated selector

If C does not beat B and compute-matched SC by either ≥2 accuracy points or ≥20% token reduction at iso-accuracy across 3 seeds with paired CI excluding zero, stop or pivot.

## Confidence

**medium**. This is the strongest evidence-backed hypothesis, not a proven result.

---

# 11. Formal Method Description

## 11.1 Problem Setup

Given input question (x), answer space (A), base model (M), strategy set (S={s_1,\dots,s_K}), and token budget (B), select an answer (\hat a) maximizing correctness under cost:

[
\max_{\pi} \mathbb{E}[\mathbf{1}(\hat a=y) - \lambda \cdot \text{tokens}]
]

where (\pi) may adaptively call strategies.

## 11.2 Existing Method Failure

CLOX-Adaptive estimates topology and chooses one fixed strategy. This fails when topology proxy is not predictive, errors are correlated, and correct candidates exist but require selection rather than more generation.

## 11.3 New Insight

The main empirical bottleneck is not “which single strategy is globally best,” but:

> Given a heterogeneous set of candidate answers, can we identify the answer cluster most likely to be correct before spending SC-level compute?

## 11.4 Method Overview

Algorithm: **CLOX-PCS**

Input:

* question (x)
* strategy portfolio (S)
* budget (B)
* calibrated selector (f_\theta)
* gate threshold (\tau)

Output:

* final answer (\hat a)
* decision trace with candidates, features, scores, costs

Steps:

1. Run low-cost scout strategies.
2. Extract normalized answers and cluster candidates.
3. Compute features for each candidate and cluster.
4. Score candidates/clusters with calibrated selector.
5. Estimate expected value of additional compute.
6. Stop if confident; otherwise call next strategy with highest expected gain/token.
7. Return best answer cluster.

## 11.5 Pseudocode

```text
Algorithm: CLOX-PCS

Input:
  x: question
  S_scout: initial low-cost strategy set
  S_expand: optional expansion strategies
  B: token budget
  f_theta: calibrated candidate correctness model
  g_phi: value-of-compute gate
  Extract: answer-type-aware extractor/checker

Output:
  final_answer, decision_log

1. H <- empty history
2. for s in S_scout:
       y_s <- RunStrategy(s, x)
       c_s <- ExtractCandidate(y_s)
       H <- H ∪ {c_s}
       if Cost(H) >= B: break

3. while Cost(H) < B:
       Phi <- FeatureExtract(H, x)
       scores <- f_theta(Phi)
       clusters <- ClusterByNormalizedAnswer(H, scores)
       best_cluster <- argmax_cluster Score(cluster)

       evi <- g_phi(best_cluster, H, remaining_budget=B-Cost(H))
       if Confidence(best_cluster) >= tau_stop or evi <= tau_cost:
           break

       s_next <- argmax_s ExpectedGainPerToken(s | H, x)
       y_next <- RunStrategy(s_next, x)
       H <- H ∪ {ExtractCandidate(y_next)}

4. final_answer <- answer(best_cluster)
5. save decision_log with raw outputs, extracted answers, features, scores, costs
6. return final_answer, decision_log
```

## 11.6 Objective

Candidate score:

[
s_\theta(c,H_x) =
\theta^\top \phi(c,H_x)
]

Cluster score:

[
S_\theta(a,H_x)
===============

## \log \sum_{c \in H_x: a_c=a} \exp(s_\theta(c,H_x))

\lambda \cdot \text{cost}(H_x)
+
\mu \cdot \text{diversity_support}(a)
]

Loss:

[
L_{\text{rank}}
===============

-\log
\frac{\sum_{c:z_c=1}\exp(s_\theta(c,H_x))}
{\sum_{c}\exp(s_\theta(c,H_x))}
]

[
L_{\text{cal}} = \frac{1}{N}\sum_c (p_\theta(c)-z_c)^2 + \text{ECE}
]

[
L_{\text{total}} =
L_{\text{rank}} + \alpha L_{\text{cal}} + \beta \max(0,\text{cost}/B-1)
]

## 11.7 Each Loss Term ↔ Phenomenon

| Term              | Addresses                                           | Why                                                            |
| ----------------- | --------------------------------------------------- | -------------------------------------------------------------- |
| `L_rank`          | oracle gap but naive vote failure                   | Learns which correct candidate to select                       |
| `L_cal`           | BAV gate non-discriminative, overconfident clusters | Forces confidence to match correctness                         |
| `L_cost`          | SC strong but expensive                             | Controls token budget                                          |
| diversity support | low-correlation pairs / candidate complementarity   | Rewards independent evidence, not repeated same-strategy votes |

## 11.8 Variables from Existing Code

| Variable                   | Existing Source                       |
| -------------------------- | ------------------------------------- |
| raw text, tokens, logprobs | `engine.GenerationOutput`             |
| strategies                 | `strategies_v2.py`                    |
| prediction/extraction      | `engine.extract_answer`, to rewrite   |
| correctness                | `evaluation.check_answer`, to rewrite |
| topology proxies           | `topology_v2.py`, rename to features  |
| stats                      | `evaluation.py` bootstrap/McNemar     |

## 11.9 New Variables Claude Code Must Add

* `candidate_id`
* `raw_output`
* `normalized_answer`
* `answer_cluster_id`
* `strategy_name`
* `sample_index`
* `selector_features`
* `selector_score`
* `cluster_score`
* `gate_decision`
* `actual_tokens`
* `calibration_split`
* `test_split`
* `run_manifest_id`

## 11.10 Required Logging

* Raw output and extracted answer for every candidate
* Answer-type-specific extraction path
* Candidate correctness on calibration/test only after labels applied
* All features before scoring
* Selector probability and calibration bin
* Gate EVI and stop/continue decision
* Token cost by strategy call
* Whether final answer came from A/B/C control
* Per-example win/loss against SC and BAV

## 11.11 Required Ablations

1. Same portfolio + majority vote.
2. Same portfolio + uncalibrated confidence.
3. Same selector without topology features.
4. Same selector without backward verifier features.
5. Same selector without value-of-compute gate.
6. Existing best positive fragment only.
7. SC-k matched tokens.
8. Oracle upper bound, clearly labeled non-deployable.

---

# 12. Related Work and Novelty Risk

Self-Consistency samples multiple reasoning paths and marginalizes/majority-votes answers; it reported large gains on GSM8K, SVAMP, AQuA, StrategyQA, and ARC-Challenge.([arXiv][19]) Tree of Thoughts explores over coherent “thought” units with self-evaluation/backtracking.([arXiv][20]) Self-Refine iteratively critiques and refines outputs without additional training.([arXiv][21]) Snell et al. study compute-optimal test-time scaling and show allocation effectiveness depends on prompt difficulty.([arXiv][22]) FOBAR combines forward and backward reasoning to verify candidate answers on math tasks.([arXiv][23]) Self-Certainty scores Best-of-N candidates using output distribution confidence and has public code.([arXiv][24]) Route-to-Reason is especially close because it dynamically allocates LMs and reasoning strategies under budget constraints.([arXiv][25])

| Paper                                        | Year / Venue        | Code                      | Mechanism                                                | Why Close                                        | Difference from New MAIN METHOD                                                             | Novelty Risk             | Required Differentiation Experiment                                                               |
| -------------------------------------------- | ------------------- | ------------------------- | -------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------- |
| Self-Consistency                             | 2022 / ICLR         | common prompting baseline | sample multiple CoT paths, majority vote                 | Current strongest baseline                       | PCS uses heterogeneous strategies + calibrated selection, not same-strategy vote            | High baseline risk       | Beat compute-matched SC and SC-k5/k8, or show iso-accuracy token saving                           |
| Tree of Thoughts                             | 2023 / NeurIPS      | official repo visible     | search over thoughts with self-evaluation                | Structured test-time compute                     | PCS does not tree-search; selects among strategy portfolio candidates                       | Medium                   | Compare at matched token budget on tasks where ToT applicable                                     |
| Self-Refine                                  | 2023                | official repo visible     | feedback/refine loop                                     | Repair/regeneration close to old CLOX            | PCS treats refine as candidate source; selector is core                                     | Medium                   | Include full_regeneration/self-refine baseline                                                    |
| Snell et al. test-time compute               | 2024/2025 ICLR Oral | likely public             | compute-optimal allocation by difficulty/verifier/search | Very close high-level framing                    | PCS focuses on cross-strategy portfolio and calibrated selector under repo’s strategy suite | High                     | Show benefit over Best-of-N and compute-optimal SC at same budget                                 |
| FOBAR                                        | 2024 Findings ACL   | public project/repo       | forward + backward verification for math                 | Backward cloze component close                   | PCS generalizes beyond 2-strategy math pair and learns selector                             | High for backward claims | Include FOBAR official reproduction on math; do not claim backward verification novel             |
| CoTnPoT                                      | 2024                | unclear/likely            | combines CoT and PoT verification                        | Cross-format strategy verification               | PCS is general K-strategy and not code-only                                                 | Medium/high              | Compare on math/code if claiming cross-strategy verification                                      |
| Self-Certainty                               | 2025                | public code               | reward-free BoN candidate scoring                        | Directly close to calibrated candidate selection | PCS must show cross-strategy features add beyond self-certainty                             | Very high                | Add self-certainty score as baseline and as feature; ablate it                                    |
| Route-to-Reason                              | 2025                | not confirmed in search   | adaptive routing over models/strategies under budget     | Closest to PCS routing                           | PCS must emphasize candidate-level outcome calibration and cross-strategy answer clustering | Very high                | Reproduce RTR or implement equivalent learned router; show PCS’s answer-cluster selector improves |
| PRISM / Plan-before-solving adaptive routing | 2026-ish            | code reportedly released  | problem-aware strategy routing                           | Similar adaptive strategy selection              | PCS routes based on generated candidate evidence, not only pre-solve problem representation | High                     | Compare pre-generation router vs post-candidate PCS                                               |
| AVA / Anytime Verified Agents                | 2026                | recent                    | adaptive compute across search/sampling/verification     | Compute gate close                               | PCS simpler reasoning-benchmark portfolio; not agent/tool framework                         | Medium                   | Show value-of-compute gate vs fixed budget                                                        |

**Novelty risk judgment:** high. The safe novelty claim is not “first adaptive strategy selector.” It must be:

> “A calibrated, answer-cluster-level framework for converting cross-strategy candidate diversity into compute-efficient reasoning, with explicit tests showing when strategy diversity adds value beyond single-strategy Best-of-N/SC and beyond self-certainty.”

Claims to avoid:

* “first strategy selector”
* “first topology theory”
* “topology determines optimal strategy”
* “masked repair is provably useful on real LLM reasoning”
* “SOTA” unless official baselines pass

---

# 13. Keep / Delete / Rewrite / Archive Plan

| Item                                  | Type     | File / Directory / Claim / Experiment | Current Role                        | Problem Under New MAIN PATH               | Action                                              | Reason                                                               |
| ------------------------------------- | -------- | ------------------------------------- | ----------------------------------- | ----------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------- |
| vLLM engine                           | Code     | `code/engine.py`                      | generation backend                  | extraction mixed in                       | REWRITE                                             | Keep generation; move answer extraction to answer-type-aware module. |
| Evaluation metrics                    | Code     | `code/evaluation.py`                  | correctness/stats                   | permissive fallback                       | REWRITE                                             | Must fix before any result.                                          |
| Benchmark loaders                     | Code     | `code/benchmarks.py`                  | data                                | silent StrategyQA train fallback, first-N | REWRITE                                             | Add split manifest, explicit source logging.                         |
| `run_full_experiment.py`              | Code     | main runner                           | legacy phases                       | seed/checkpoint/delete issues             | REWRITE                                             | Preserve as baseline runner but add manifest/raw JSONL.              |
| `strategies_v2.py`                    | Code     | candidate strategies                  | some prompts not mechanism-faithful | KEEP ONLY AS BASELINE / GENERATOR         | Strategies become candidate generators, not claims. |                                                                      |
| `UncertaintyTargetedRepair`           | Method   | `strategies_v2.py`                    | old positive fragment               | full rewrite confound                     | KEEP ONLY AS ABLATION                               | Use to test “old fragment only”; rewrite strict version separately.  |
| `RandomRepair`                        | Ablation | `strategies_v2.py`                    | negative control                    | unstable hash                             | REWRITE                                             | Needed as control with stable seed.                                  |
| `BackwardCloze`                       | Method   | `strategies_v2.py`                    | backward candidate                  | close to FOBAR, compute issues            | KEEP ONLY AS ABLATION / FEATURE                     | Do not claim novelty; use as verifier feature.                       |
| `CLOXAdaptive`                        | Method   | `strategies_v2.py`                    | old main method                     | topology threshold failed                 | FREEZE                                              | Legacy baseline only.                                                |
| `topology_v2.py`                      | Code     | topology                              | theoretical estimator               | not validated                             | REWRITE / MERGE INTO NEW METHOD                     | Rename as proxy feature extractor.                                   |
| `results/v3`, `results/v4`            | Results  | topology/pilot                        | evidence                            | inconsistent / old model                  | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE           | Useful for phenomenon ledger, not main proof.                        |
| `results/bav`                         | Results  | BAV                                   | positive efficiency signal          | not SC-beating                            | KEEP ONLY AS BASELINE                               | Compare PCS vs BAV.                                                  |
| `results/meta`                        | Results  | oracle gap                            | opportunity                         | oracle label-using                        | KEEP ONLY AS HISTORICAL POSITIVE SIGNAL             | Cannot report as method.                                             |
| `EXPERIMENTS.md`                      | Docs     | early failed experiment               | audit trail                         | contaminated positives                    | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE           | Important integrity record.                                          |
| `RESEARCH_BRIEF.md`                   | Docs     | failure lessons                       | v3 motivation                       | overambitious “best paper”                | KEEP / WEAKEN CLAIMS                                | Keep lessons; remove hype.                                           |
| `IDEA_REPORT.md`                      | Docs     | idea A                                | cross-strategy                      | naive voting now contradicted by pilot    | ARCHIVE + EXTRACT INTO PCS                          | Use diversity insight, not recommendation as-is.                     |
| `paper/main.tex`                      | Paper    | current thesis                        | overclaims                          | unsupported                               | REWRITE                                             | New thesis PCS.                                                      |
| `paper` theorem claims                | Claim    | masking/topology theorem              | old novelty                         | not empirical                             | ARCHIVE                                             | Maybe appendix if clearly toy theory.                                |
| `docs/novelty_report.json`            | Docs     | novelty                               | likely stale                        | contradicts web                           | ARCHIVE                                             | Manual related work required.                                        |
| Legacy `strategies.py`, `topology.py` | Code     | old implementation                    | dead/confusing                      | not main path                             | ARCHIVE                                             | Reduce confusion.                                                    |
| Plot scripts                          | Code     | analysis                              | useful                              | schema mismatch                           | REWRITE                                             | Support PCS result schema.                                           |
| Failed folders                        | Results  | historical                            | clutter                             | evidence                                  | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE           | Do not delete silently.                                              |

---

# 14. Claude Code Implementation Plan

## Task 1: Freeze and isolate legacy paths

Purpose: prevent old topology/repair claims from contaminating new PCS implementation.
Which Phenomenon / Constraint It Addresses: P02, P05, P06, C01, C03.
Why It Supports New MAIN METHOD PATH: PCS must be validated against old paths, not mixed with them.
Files to Inspect: `code/strategies_v2.py`, `code/run_full_experiment.py`, `paper/main.tex`, `README.md`.
Files to Edit: `README.md`, add `docs/legacy_status.md`.
Files to Delete / Archive: move old experimental notes only via `archive/legacy_clox_v2/`, do not delete.
Functions / Classes: `CLOXAdaptive`, `UncertaintyTargetedRepair`, old topology selector.
Exact Change: mark CLOXAdaptive/topology-threshold as `LEGACY_BASELINE`; add warning that PCS is new main path.
Do Not Change: do not change strategy outputs yet.
Verification Command: `python -m compileall code`
Expected Result: no behavior change; docs clarify legacy status.
Failure Means: accidental code break or docs mismatch.
Rollback Condition: compile fails.
Priority: P0.
Confidence: high.

## Task 2: Add immutable run manifest and result schema

Purpose: stop losing raw evidence.
Which Phenomenon / Constraint It Addresses: checkpoint deletion, C11.
Files to Inspect: `run_full_experiment.py`, `analyze_*.py`.
Files to Edit: new `code/result_schema.py`; edit `run_full_experiment.py`.
Exact Change: save `manifest.json`, `per_example/*.jsonl`, `candidate_outputs/*.jsonl`; never delete raw checkpoints without archive.
Do Not Change: strategy logic.
Verification Command: `python code/smoke_test.py --max_examples 2` or add `python code/run_full_experiment.py --phase strategies --benchmarks gsm8k --max_examples 2 --seeds 11 --output results/smoke_schema`
Expected Result: manifest + raw JSONL present.
Failure Means: no auditable result.
Rollback Condition: schema breaks existing analysis.
Priority: P0.
Confidence: high.

## Task 3: Fix answer extraction and checking

Purpose: remove metric contamination.
Which Phenomenon / Constraint It Addresses: P15, C07.
Files to Inspect: `engine.py`, `evaluation.py`, `benchmarks.py`.
Files to Edit: new `code/answer_extraction.py`; edit `evaluation.py`, `engine.py`.
Exact Change: implement answer-type-aware extraction/checking; MC compare labels only; boolean compare yes/no only; numeric compare parsed final numeric; remove generic substring fallback.
Do Not Change: benchmark answers except normalization.
Verification Command: `pytest tests/test_answer_extraction.py`
Expected Result: known bad ARC/text cases fail correctly.
Failure Means: all historical metrics remain suspect.
Rollback Condition: official benchmark examples fail due parser too strict; then add benchmark-specific parser, not fallback.
Priority: P0.
Confidence: high.

## Task 4: Fix seed and sampling reproducibility

Purpose: make multi-seed real.
Which Phenomenon / Constraint It Addresses: seed audit, C02.
Files to Inspect: `engine.py`, `run_full_experiment.py`, `strategies_v2.py`.
Files to Edit: `code/utils.py`, `run_full_experiment.py`, `strategies_v2.py`.
Exact Change: `set_global_seed(seed)`; stable hash via SHA256; pass seed to strategies; log vLLM seed/sampling params.
Do Not Change: model, prompt, dataset.
Verification Command: run same seed twice on 3 examples and diff JSONL; run different seeds and confirm stochastic methods differ when temperature >0.
Expected Result: reproducible same-seed results.
Failure Means: vLLM seed not controlled; must document nondeterminism.
Rollback Condition: deterministic baseline changes unexpectedly.
Priority: P0.
Confidence: high.

## Task 5: Add dataset split manifest

Purpose: prevent first-N bias and leakage.
Which Phenomenon / Constraint It Addresses: pilot collapse, StrategyQA fallback.
Files to Inspect: `benchmarks.py`.
Files to Edit: `benchmarks.py`, new `code/split_manifest.py`, `configs/splits/*.json`.
Exact Change: log dataset repo/split/fingerprint; create seeded random or stratified fixed IDs; forbid train split for eval by default.
Do Not Change: benchmark content.
Verification Command: `python code/split_manifest.py --benchmarks gsm8k,math,strategyqa,arc_challenge --n_calib 50 --n_test 100 --seed 11`
Expected Result: manifests saved and reused.
Failure Means: evaluation not reproducible.
Rollback Condition: dataset unavailable; require explicit dev-mode.
Priority: P0.
Confidence: high.

## Task 6: Implement portfolio candidate generator

Purpose: PCS candidate collection.
Which Phenomenon / Constraint It Addresses: P09, P10, P13.
Files to Inspect: `strategies_v2.py`, `engine.py`.
Files to Edit: new `code/portfolio.py`.
Exact Change: run configured strategies; collect raw output, normalized answer, cost, strategy, confidence, logprob summaries.
Do Not Change: baseline strategy internals.
Verification Command: `python code/run_portfolio_experiment.py --mode collect --benchmark gsm8k --split calib --max_examples 3 --strategies standard_cot,compute_matched_sc,backward_cloze,targeted_repair`
Expected Result: candidate JSONL with ≥1 candidate/example.
Failure Means: PCS cannot be evaluated.
Rollback Condition: candidate collection changes baseline outputs.
Priority: P1.
Confidence: high.

## Task 7: Implement calibrated selector

Purpose: add missing outcome-aware mechanism.
Which Phenomenon / Constraint It Addresses: P09-P13, C04-C08.
Files to Inspect: new candidate JSONL.
Files to Edit: new `code/calibrated_selector.py`, `code/features.py`.
Exact Change: train logistic/isotonic selector on calibration set; save model artifact and feature schema; compute ECE/Brier.
Do Not Change: test labels during training.
Verification Command: `python code/calibrated_selector.py fit --calib results/pcs/calib_candidates.jsonl --out results/pcs/selector.pkl`
Expected Result: selector saved; calibration report generated.
Failure Means: feature pipeline broken or no signal.
Rollback Condition: leakage detected.
Priority: P1.
Confidence: medium.

## Task 8: Implement value-of-compute gate

Purpose: decide stop/continue under budget.
Which Phenomenon / Constraint It Addresses: BAV gate failure, SC cost.
Files to Edit: `code/compute_gate.py`, `code/run_portfolio_experiment.py`.
Exact Change: first implement conservative gate: stop if cluster calibrated confidence ≥τ; else expand one configured strategy if remaining budget.
Do Not Change: budget accounting.
Verification Command: `python code/run_portfolio_experiment.py --mode eval --budget_tokens 8000 --gate conservative`
Expected Result: logs stop/continue decisions.
Failure Means: no adaptive compute behavior.
Rollback Condition: token budget violated.
Priority: P1.
Confidence: medium.

## Task 9: Add A/B/C control evaluation

Purpose: prove PCS is not old positive fragment.
Files to Edit: `code/run_portfolio_experiment.py`, `code/analyze_pcs.py`.
Exact Change: evaluate A Existing Best Positive Fragment Only, B same portfolio without calibrated selector, C full PCS.
Verification Command: `python code/analyze_pcs.py --runs results/pcs/minimal --compare A,B,C --paired`
Expected Result: per-example win/loss, bootstrap CI, token Pareto.
Failure Means: cannot support new method.
Rollback Condition: A/B/C not same split/budget.
Priority: P0.
Confidence: high.

## Task 10: Rewrite paper thesis only after minimal tests pass

Purpose: avoid overclaiming.
Files to Edit: `paper/main.tex`, `README.md`.
Exact Change: replace topology theorem main claim with PCS thesis; move old topology to appendix/negative lessons.
Verification Command: `latexmk -pdf paper/main.tex` if available.
Expected Result: paper no longer claims unsupported near-oracle topology routing.
Failure Means: academic integrity risk.
Rollback Condition: experiments fail; keep paper as draft with negative report.
Priority: P2.
Confidence: high.

---

# 15. Minimal Verification Experiments

| Priority | Experiment                                | Hypothesis                                                      | Command                                                                                  | Config                     | Dataset            | Seeds    | Metric            | Success Criterion                                             | Failure Interpretation              |
| -------: | ----------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------- | ------------------ | -------- | ----------------- | ------------------------------------------------------------- | ----------------------------------- |
|        0 | Smoke test                                | Code imports and minimal generation work                        | `python -m compileall code && pytest tests/test_answer_extraction.py`                    | N/A                        | synthetic/unit     | N/A      | pass/fail         | all pass                                                      | stop; code not auditable            |
|        0 | Data sanity check                         | Fixed split manifests are valid                                 | `python code/split_manifest.py --check`                                                  | `configs/splits/*.json`    | all target         | 11       | IDs/source/split  | no train eval unless dev flag                                 | leakage risk                        |
|        0 | Metric sanity check                       | MC/boolean/numeric metrics are strict                           | `pytest tests/test_metrics.py`                                                           | N/A                        | fixtures           | N/A      | pass/fail         | all expected cases pass                                       | old metrics unusable                |
|        0 | One-batch overfit/calibration sanity      | Selector can learn trivial signal on tiny labeled set           | `python code/calibrated_selector.py fit --debug_tiny`                                    | debug                      | 10 calib           | 11       | train AUC/Brier   | AUC >0.9 on toy                                               | feature/label bug                   |
|        0 | Checkpoint loading check                  | Raw outputs persist and resume safe                             | run 2 examples, interrupt/resume                                                         | smoke                      | GSM8K              | 11       | JSONL consistency | no duplicate/loss                                             | result schema broken                |
|        1 | Reproduce current negative result         | targeted≈random / vote<SC on pilot                              | `python code/analyze_pilot.py --input results/pilot/pilot_analysis.json`                 | legacy                     | pilot              | existing | acc/tokens        | reproduces summary                                            | old diagnostics unreliable          |
|        1 | Reproduce current best positive fragment  | BAV Pareto / targeted pilot                                     | `python code/analyze_bav.py --input results/bav`                                         | legacy                     | pilot              | existing | acc/tokens        | matches BAV 0.80                                              | old result unreadable               |
|        1 | New mechanism activation check            | Selector scores vary and calibrate                              | `python code/calibrated_selector.py report`                                              | PCS calib                  | calib split        | 11       | ECE/Brier/AUC     | AUC >0.55, ECE < heuristic                                    | no selection signal                 |
|        1 | PCS minimal test                          | Full PCS improves over uncalibrated portfolio                   | `python code/run_portfolio_experiment.py --mode eval --compare A,B,C --max_examples 100` | `configs/pcs_minimal.yaml` | GSM8K or mixed     | 11,23,37 | acc/tokens/CI     | C > B and compute-matched SC by ≥2 pts or iso-acc -20% tokens | main path weak                      |
|        1 | Key ablation: remove new mechanism        | Same candidates + majority vote                                 | same as above                                                                            | `selector=majority`        | same               | same     | acc/tokens        | lower than C                                                  | if equal, selector not contributing |
|        1 | Key ablation: existing best fragment only | BAV or targeted only                                            | same                                                                                     | `mode=A`                   | same               | same     | acc/tokens        | lower than C                                                  | if A wins, PCS unnecessary          |
|        1 | New method without gate                   | calibrated selector fixed full budget                           | same                                                                                     | `gate=off`                 | same               | same     | acc/tokens        | accuracy maybe similar, tokens higher                         | if better and cheap, gate wrong     |
|        1 | Full new method                           | PCS selector + gate                                             | same                                                                                     | `gate=on`                  | same               | same     | acc/tokens        | Pareto improvement                                            | if not, stop/pivot                  |
|        2 | Small baseline comparison                 | PCS vs CoT/SC/compute-matched SC                                | `python code/analyze_pcs.py --baselines`                                                 | minimal                    | GSM8K/MATH/ARC     | 3 seeds  | paired CI         | not worse than SC at matched budget                           | no paper claim                      |
|        2 | Multi-seed stability                      | effect not seed artifact                                        | same                                                                                     | minimal                    | same               | 11,23,37 | mean±std          | std acceptable; CI >0                                         | instability                         |
|        2 | Expansion gate                            | decide full benchmark                                           | `python code/analyze_pcs.py --gate_decision`                                             | minimal                    | all                | 3 seeds  | criteria          | passes pre-registered gate                                    | do not scale                        |
|        2 | Official baseline reproduction            | compare to official/known SC/Self-Certainty/FOBAR               | separate scripts                                                                         | baseline configs           | task-specific      | 3        | acc/tokens        | within expected range                                         | unfair comparison                   |
|        2 | Unified environment comparison            | all methods same model/env/split                                | run manifest                                                                             | full env                   | all                | 3        | manifest diff     | identical model/dataset                                       | invalid                             |
|        3 | Robustness/generalization                 | PCS trained on calib generalizes to held-out tasks/difficulties | `--split test_hard`                                                                      | robustness                 | hard subsets / BBH | 3        | acc/tokens        | no collapse                                                   | selector overfits                   |
|        3 | Statistical significance                  | effect survives paired tests                                    | `python code/analyze_pcs.py --paired_bootstrap --mcnemar`                                | final                      | all                | 3–5      | CI/p              | CI excludes 0 for main claim                                  | report as inconclusive              |

Core controls:

* **A. Existing Best Positive Fragment Only:** BAV/cross-strategy heuristic or targeted_repair, pre-registered by calibration dev performance but frozen before test.
* **B. New MAIN METHOD Without New Mechanism:** same candidate portfolio, no calibrated selector/gate; majority vote or raw confidence.
* **C. Full New MAIN METHOD:** calibrated selector + value-of-compute gate.

---

# 16. Baseline and SOTA Plan

| Baseline                    | Why Required                    | Official Code                 | Dataset        | Metric                 | Reproduction Requirement                                   | Fairness Risk                    |
| --------------------------- | ------------------------------- | ----------------------------- | -------------- | ---------------------- | ---------------------------------------------------------- | -------------------------------- |
| Standard CoT                | simplest baseline               | prompt implementation         | all            | exact/official         | same prompts, same split                                   | too weak alone                   |
| Self-Consistency k=3/5/8    | strongest local baseline        | known method                  | all            | acc/tokens             | same model/temp/budget                                     | must not under-sample            |
| Compute-matched SC          | cost fairness                   | internal                      | all            | acc/tokens             | token-matched to PCS budget                                | token accounting must be actual  |
| Full SC / Best-of-N         | upper compute baseline          | internal/official variants    | all            | acc/tokens             | fixed K and tokens                                         | may dominate if budget unlimited |
| Self-Certainty              | closest candidate scorer        | public code found             | math/reasoning | acc/tokens/ECE         | official score or faithful reproduction                    | novelty risk high                |
| FOBAR                       | backward verification close     | public project/repo found     | math           | acc/tokens             | official or faithful reproduction                          | API/model mismatch               |
| CoTnPoT                     | cross-format verification       | check official availability   | math/code      | acc                    | if used as related baseline                                | task mismatch                    |
| Route-to-Reason / RTR       | adaptive strategy routing close | not confirmed in quick search | reasoning      | acc/tokens             | reproduce if code available, else strong internal analogue | novelty risk very high           |
| PRISM/problem-aware routing | adaptive routing close          | code reportedly released      | math/reasoning | acc/tokens             | official reproduction if accessible                        | may subsume routing claim        |
| Existing CLOXAdaptive       | legacy baseline                 | internal                      | all            | acc/tokens             | frozen code after metric fix                               | must not be silently improved    |
| BAV                         | existing positive fragment      | internal                      | pilot/full     | acc/tokens             | exact old heuristic and fixed splits                       | if tuned on test, invalid        |
| Targeted/random repair      | ablation controls               | internal                      | all            | acc/tokens/wrong→right | strict repaired version + legacy                           | old prompt confounds             |

---

# 17. Paper Thesis Reconstruction

1. **New Paper Thesis:**
   Test-time reasoning gains are bottlenecked not by generating more traces alone, but by selecting among heterogeneous strategy-generated candidates under a compute budget. A calibrated answer-cluster selector can convert cross-strategy diversity into reliable accuracy/token gains when candidate errors are not fully correlated.

2. **Main Technical Contribution:**
   CLOX-PCS: a candidate-level, calibrated, value-of-compute-controlled portfolio framework for inference-time reasoning.

3. **Main Empirical Claim:**
   If experiments pass: PCS improves the accuracy–token Pareto frontier over same-budget SC, BAV/cross-vote, and existing repair fragments on held-out splits.

4. **What Previous Failures Taught Us:**
   Topology proxies alone do not route reliably; masking is not automatically causal; pilot positives collapse; diversity without selection underperforms SC.

5. **What We Should Not Claim:**
   Do not claim topology determines optimal strategy, masked repair is proven useful on real LLMs, SOTA, or first adaptive routing.

6. **What We Can Claim If Experiments Pass:**
   Cross-strategy candidate diversity contains exploitable signal; calibrated selection/gating is necessary; PCS outperforms uncalibrated portfolios and existing positive fragments under matched budgets.

7. **Required Baselines:**
   SC, compute-matched SC, Self-Certainty, FOBAR on math, RTR/PRISM-like router if available, BAV, legacy CLOXAdaptive.

8. **Required Ablations:**
   no selector, no gate, no topology features, no backward verifier, same-strategy-only portfolio, random strategy portfolio, old best fragment only.

9. **Required Robustness Tests:**
   held-out tasks, hard subsets, different model sizes, MC/numeric/boolean answer types, calibration transfer.

10. **Reviewer Likely Objections:**
    “This is just routing,” “Self-Certainty already scores candidates,” “baselines weak,” “selector trained on test,” “metric permissive,” “oracle gap cherry-pick.”

11. **How New MAIN METHOD Answers Them:**
    Strict split manifests, official baselines, A/B/C controls, raw per-example logs, calibration metrics, cross-strategy-specific ablations.

12. **What Would Make This NeurIPS-Strong:**
    Clear Pareto improvements across ≥3 tasks and ≥2 models, robust baselines, mechanism analysis showing selector converts oracle gap, negative results reported.

13. **What Would Make This Rejected:**
    Only beats weak baselines, only works on first-N GSM8K, no official Self-Certainty/RTR comparison, metric bugs remain, claim overreach.

14. **What Would Be Required for Oral-Level Strength:**
    Broad generalization, simple theory predicting when PCS beats SC, strong compute savings, public reproducible benchmark suite.

15. **What Would Be Required for Best-Paper-Level Strength:**
    A genuinely general law of strategy diversity/selection with strong theoretical and empirical support across models/tasks; current evidence is far from that.

---

# 18. Reviewer Risk Assessment

| Risk                       | Why Reviewer May Object                         | Evidence Needed                     | How New MAIN METHOD Addresses It                         | Remaining Weakness                     |
| -------------------------- | ----------------------------------------------- | ----------------------------------- | -------------------------------------------------------- | -------------------------------------- |
| Novelty risk               | Adaptive routing/BoN/self-certainty are crowded | Related work + official comparisons | PCS focuses on cross-strategy answer-cluster calibration | Risk remains high                      |
| Incremental risk           | Could look like engineering wrapper             | Mechanism ablations                 | A/B/C controls isolate selector/gate                     | Needs clean theory                     |
| Baseline weakness          | SC/Self-Certainty/RTR may dominate              | Official/faithful baselines         | Baseline plan includes them                              | Compute may be high                    |
| Reproducibility            | Existing seeds/checkpoints weak                 | manifests/raw JSONL                 | Tasks 2–5 fix this                                       | Need execution                         |
| Cherry-picking             | Pilot positives unstable                        | fixed splits/multiseed              | Pre-registered success criteria                          | Must report failures                   |
| Negative hiding            | Repo has many failures                          | negative ledger in paper appendix   | Keep historical evidence                                 | Narrative risk                         |
| Overclaiming               | Current paper overclaims near-oracle            | rewrite thesis                      | Claim only after tests                                   | Must delete old abstract claims        |
| Unclear mechanism          | Selector could be black box                     | feature ablations/calibration       | log features/scores/EVI                                  | If model too complex, reviewers object |
| Ablation insufficiency     | Need prove not old fragment                     | A/B/C controls                      | explicit experiments                                     | Must be strict                         |
| Dataset limitation         | StrategyQA/ARC issues                           | diverse tasks + official metrics    | split by answer type                                     | Some tasks saturated                   |
| Compute unfairness         | SC costs more/less                              | actual token accounting             | cost logs and budgets                                    | vLLM token accounting must be exact    |
| Implementation reliability | Metric/extraction bugs                          | unit tests                          | Task 3                                                   | Old results remain low-confidence      |
| Related work omission      | Many close works                                | manual audit                        | Section 12                                               | Must stay current                      |

---

# 19. Final Decision

## 1. One-Sentence Verdict

从所有正面、负面、不稳定结果综合看，最值得验证的新主线是 **CLOX-PCS: a calibrated cross-strategy candidate portfolio selector with value-of-compute gating**，而不是继续推进 topology-threshold CLOXAdaptive 或 targeted repair。

## 2. Current Most Likely Root Cause

当前失败最可能来自组合原因：

* **method assumption failure:** topology proxy does not determine optimal strategy.
* **missing mechanism:** no calibrated selector/gate to exploit candidate diversity.
* **evaluation/code reliability issues:** answer checking, seed control, checkpoint/raw result preservation.
* **weak experimental setup:** first-N pilot, missing full comparison, missing proxy validation.
* **novelty risk:** adaptive routing and candidate scoring are already crowded.

## 3. Why This Is Not Just the Existing Best Path

PCS 不选择 targeted_repair、BAV 或 IDEA A naive cross-vote 作为最终路线。它把这些都降级为 evidence fragments / candidate generators，新增的核心机制是 **held-out calibrated answer-cluster selection + value-of-compute control**。

## 4. Phenomena Explained

PCS explains:

* targeted pilot positive but scale collapse;
* targeted=random negative;
* topology correlation near zero;
* BAV efficiency but weak gate;
* oracle gap without deployable gain;
* SC dominance under naive voting;
* low-correlation pairs insufficient for majority vote.

## 5. Mechanism Missing in Current Method

缺少 **outcome-aware calibrated selection**：旧方法只生成候选或按固定规则路由，不会可靠判断候选正确性。

## 6. New Mechanism

Calibrated selector learns:

[
P(\text{candidate or answer cluster correct} \mid \text{strategy, support, confidence, verifier, topology proxy, cost})
]

and gate estimates whether more compute is worth spending.

## 7. What to Delete / Archive / Rewrite

* **Rewrite:** `evaluation.py`, `engine.extract_answer`, `benchmarks.py`, `run_full_experiment.py` result schema.
* **Freeze:** `CLOXAdaptive` as legacy baseline.
* **Archive:** old topology theorem main paper claim, stale novelty report, legacy `strategies.py/topology.py`.
* **Keep as ablation:** targeted repair, random repair, backward cloze, BAV.
* **Keep as historical negative evidence:** `EXPERIMENTS.md`, v2/v3 failure notes.

## 8. First Five Claude Code Tasks

1. Freeze legacy CLOXAdaptive/topology route.
2. Add immutable result schema and run manifest.
3. Fix answer extraction/checking with tests.
4. Fix seed control and stable random repair.
5. Add split manifests and explicit dataset source logging.

## 9. Minimal Experiments

1. metric/data/checkpoint sanity;
2. reproduce old negative and best positive fragments;
3. collect PCS calibration candidates;
4. fit selector;
5. compare A/B/C on held-out test:

   * A existing best fragment only,
   * B same portfolio without selector,
   * C full PCS.

## 10. Continue / Stop / Pivot Criteria

Continue if:

* C beats A and B on held-out split;
* C beats compute-matched SC by ≥2 points or saves ≥20% tokens at iso-accuracy;
* paired CI excludes zero on at least core benchmark set;
* metric/seed/split tests pass.

Stop if:

* selector AUC ≤0.55 and no token saving;
* C ≈ B, meaning calibration adds nothing;
* SC dominates at same budget;
* answer metric fixes erase all positive signal.

Pivot if:

* candidate oracle gap remains high but selector fails: pivot to stronger verifier / self-certainty integration.
* candidate oracle gap disappears after metric fixes: abandon cross-strategy PCS and write negative/diagnostic report.

## 11. NeurIPS-Level Gap

Need official baselines, fixed evaluation, held-out multi-seed results, robust ablations, and a narrower honest thesis.

## 12. Oral / Best Paper Gap

Oral-level would require broad, stable Pareto improvements and a clean theory of when cross-strategy selection beats SC. Best-paper-level would require a field-shifting law of test-time compute allocation; current evidence is not there.

## 13. Confidence

**medium**. Evidence against old route is high-confidence. Evidence that PCS is the best next route is medium-confidence because it is a falsifiable hypothesis supported by oracle gap/diversity signals, not yet proven.

---

# 20. Final Claude Code Instruction

```text
Claude Code, execute the following plan.

You must implement the New MAIN METHOD PATH defined in the GPT-5.5 Pro diagnosis report:

CLOX-PCS: Calibrated Portfolio Compute Selection.

Do not invent a different method.
Do not optimize for superficial positive results.
Do not weaken baselines.
Do not delete negative results silently.
Do not change metrics or datasets unless explicitly instructed.
Do not rewrite unrelated files.
Do not treat targeted_repair, BAV, topology routing, or cross-strategy majority vote as the final method.
They are baselines, ablations, or historical evidence only.

Your tasks are:

1. Freeze legacy CLOXAdaptive/topology-threshold routing.
   - Mark it as LEGACY_BASELINE.
   - Do not modify its behavior except documentation and explicit labeling.
   - Add docs/legacy_status.md explaining that topology routing is not the new main path.

2. Add immutable result schema and run manifests.
   - Create code/result_schema.py.
   - Save manifest.json for every run.
   - Save per-example raw JSONL for every strategy/seed.
   - Never delete .ckpt or raw predictions unless archived with a manifest.
   - Modify run_full_experiment.py accordingly.

3. Fix answer extraction and evaluation.
   - Create code/answer_extraction.py.
   - Rewrite evaluation.check_answer to be answer-type-specific.
   - Multiple-choice must compare extracted option labels only.
   - Boolean must compare yes/no only.
   - Numeric/math must parse final numeric/math answer.
   - Remove generic substring fallback.
   - Add tests/test_answer_extraction.py and tests/test_metrics.py.

4. Fix reproducibility.
   - Add set_global_seed(seed).
   - Pass seeds into stochastic strategies.
   - Replace Python hash(question) with stable SHA256-based seeded hashing.
   - Log vLLM seed, temperature, top_p, max_tokens, model path, dataset split, and strategy config.

5. Add dataset split manifests.
   - Create code/split_manifest.py.
   - Save fixed calibration/test IDs for GSM8K, MATH, StrategyQA, ARC, and optional BBH.
   - Log dataset repo, split, fingerprint, and sample IDs.
   - Forbid train split evaluation unless --allow_train_eval is explicitly set.

6. Implement PCS candidate collection.
   - Create code/portfolio.py.
   - Use existing strategies as candidate generators:
     standard_cot, compute_matched_sc, self_consistency, backward_cloze, targeted_repair, random_repair, full_regeneration, BAV if available.
   - For every candidate, save:
     raw_output, normalized_answer, strategy, sample_index, tokens, confidence/logprob summaries, answer_cluster_id, topology_proxy_features if available.

7. Implement calibrated selector.
   - Create code/features.py and code/calibrated_selector.py.
   - Fit only on calibration split.
   - Save selector artifact and feature schema.
   - Report AUC, Brier score, ECE, calibration bins.
   - Do not use test labels during training or threshold tuning.

8. Implement value-of-compute gate.
   - Create code/compute_gate.py.
   - Start with conservative gate:
     stop if best answer-cluster calibrated confidence >= tau_stop;
     otherwise call next configured strategy only if remaining budget allows.
   - Log every stop/continue decision and expected value estimate.

9. Implement A/B/C evaluation.
   - A = Existing Best Positive Fragment Only.
   - B = same portfolio without calibrated selector/gate, using majority vote or existing heuristic.
   - C = Full CLOX-PCS with calibrated selector and value-of-compute gate.
   - Use the same held-out split, model, budget, and seeds.
   - Report paired bootstrap, McNemar, per-example win/loss, accuracy, tokens, tokens-per-correct, ECE.

10. Run minimal verification only.
   - Do not run full benchmark until smoke, metric, seed, split, checkpoint, and A/B/C minimal tests pass.
   - Minimal command set:
     python -m compileall code
     pytest tests/test_answer_extraction.py tests/test_metrics.py
     python code/split_manifest.py --benchmarks gsm8k,math,strategyqa,arc_challenge --n_calib 50 --n_test 100 --seed 11
     python code/run_portfolio_experiment.py --mode collect --config configs/pcs_minimal.yaml --split calib
     python code/calibrated_selector.py fit --calib results/pcs/calib_candidates.jsonl --out results/pcs/selector.pkl
     python code/run_portfolio_experiment.py --mode eval --config configs/pcs_minimal.yaml --split test --compare A,B,C --seeds 11,23,37
     python code/analyze_pcs.py --runs results/pcs/minimal --paired --compare A,B,C

For every task:
  - make the smallest necessary change;
  - show the diff;
  - run the specified verification command;
  - save logs;
  - report failures;
  - stop if verification fails;
  - do not proceed to full benchmark until minimal tests pass.

At the end, output:
  - files changed;
  - files archived;
  - configs added;
  - commands run;
  - logs;
  - result table;
  - failed checks;
  - unresolved issues;
  - whether Full New MAIN METHOD beats:
    A. Existing Best Positive Fragment Only,
    B. New MAIN METHOD Without New Mechanism,
    C. Full New MAIN METHOD.
```

[1]: https://github.com/Sunshine535/nips-clox "https://github.com/Sunshine535/nips-clox"
[2]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/paper/main.tex "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/paper/main.tex"
[3]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/EXPERIMENTS.md "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/EXPERIMENTS.md"
[4]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/RESEARCH_BRIEF.md "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/RESEARCH_BRIEF.md"
[5]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/IDEA_REPORT.md "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/IDEA_REPORT.md"
[6]: https://github.com/Sunshine535/nips-clox/blob/main/code/evaluation.py?plain=1 "nips-clox/code/evaluation.py at main · Sunshine535/nips-clox · GitHub"
[7]: https://github.com/Sunshine535/nips-clox/blob/main/code/benchmarks.py?plain=1 "https://github.com/Sunshine535/nips-clox/blob/main/code/benchmarks.py?plain=1"
[8]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/bav/bav_analysis.json "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/bav/bav_analysis.json"
[9]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/meta/meta_summary.json "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/meta/meta_summary.json"
[10]: https://github.com/Sunshine535/nips-clox/blob/main/code/strategies_v2.py?plain=1 "nips-clox/code/strategies_v2.py at main · Sunshine535/nips-clox · GitHub"
[11]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/v3/Qwen2.5-32B-Instruct-AWQ/topology_summary.json "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/v3/Qwen2.5-32B-Instruct-AWQ/topology_summary.json"
[12]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/v4/Qwen3-8B/topology_summary.json "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/v4/Qwen3-8B/topology_summary.json"
[13]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/v3/Qwen2.5-32B-Instruct-AWQ/pilot/pilot_results.json "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/v3/Qwen2.5-32B-Instruct-AWQ/pilot/pilot_results.json"
[14]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/pilot/pilot_analysis.json "https://raw.githubusercontent.com/Sunshine535/nips-clox/main/results/pilot/pilot_analysis.json"
[15]: https://github.com/Sunshine535/nips-clox/blob/main/code/engine.py?plain=1 "nips-clox/code/engine.py at main · Sunshine535/nips-clox · GitHub"
[16]: https://github.com/Sunshine535/nips-clox/blob/main/code/run_full_experiment.py?plain=1 "nips-clox/code/run_full_experiment.py at main · Sunshine535/nips-clox · GitHub"
[17]: https://github.com/Sunshine535/nips-clox/blob/main/code/topology_v2.py?plain=1 "nips-clox/code/topology_v2.py at main · Sunshine535/nips-clox · GitHub"
[18]: https://raw.githubusercontent.com/Sunshine535/nips-clox/main/code/engine.py "raw.githubusercontent.com"
[19]: https://arxiv.org/abs/2203.11171 "https://arxiv.org/abs/2203.11171"
[20]: https://arxiv.org/abs/2305.10601 "https://arxiv.org/abs/2305.10601"
[21]: https://arxiv.org/abs/2303.17651 "https://arxiv.org/abs/2303.17651"
[22]: https://arxiv.org/abs/2408.03314 "https://arxiv.org/abs/2408.03314"
[23]: https://arxiv.org/abs/2308.07758 "https://arxiv.org/abs/2308.07758"
[24]: https://arxiv.org/html/2502.18581v2 "https://arxiv.org/html/2502.18581v2"
[25]: https://arxiv.org/abs/2505.19435 "https://arxiv.org/abs/2505.19435"
