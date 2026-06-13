# CSER 修改保留与 MSRVTT10K 全量实验前调整建议

**依据实验**: Slurm job `9805979`
**当前完成阶段**: CSER phase1 + phase2 已完成，phase3 仍在运行中
**报告根目录**: `reports/full/9805979/cser/`
**适用范围**: CSER 逻辑、实验协议、MSRVTT10K 全量实验准备

## 1. 总体判断

本次更新是正向的，但正向点主要体现在**实验机制和指标可信度**，不是检索性能提升。

已经确认的正向变化是：CSER 的 safety path 从“只报告 coverage、不真正过滤候选”变成了真实的候选过滤机制。现在 Phase2 E6 能测出 `GT_filtered_rate`、平均保留候选数和候选缩减率，因此可以真实评估 safety gate 的作用。

仍未解决的是：CSER 的 learned selector 没有提高检索质量。Phase2 主结果中 CSER 仍为 `R@1=28.0%`、`R@5=57.6%`，与 UCB 的 `R@1=28.0%` 打平，仍低于 oracle `R@1=35.2%`。Phase1 也显示 budget 5 下 learned greedy value `0.408` 低于 semantic-only value `0.427`。

因此，下次 MSRVTT10K 全量实验前，应保留本次的 safety / reporting 改动，但必须补上 10K 可扩展缓存、候选 top-k sweep、SVN selector 诊断和真实 latency 协议。

## 2. 本次结果证据

### 2.1 Phase1

| 指标 | 结果 | 判断 |
|---|---:|---|
| monotonicity violation | `17.425%` | 仍然较高，不能直接声称 monotone submodular |
| submodularity violation | `4.300%` | 满足 `<5%` 的 submodularity violation 阈值 |
| gamma mean | `0.9748` | 弱子模指标较好 |
| gamma p10 | `0.9675` | 保守分位也较好 |
| verdict | `submodular` | 仅针对 submodularity violation，不覆盖 monotonicity 问题 |

budget sweep:

| Budget | Greedy value | Oracle value | Semantic-only value | Greedy / oracle | Avg experts |
|---:|---:|---:|---:|---:|---:|
| 3.0 | `0.4119` | `0.4743` | `0.4271` | `86.9%` | `1.968` |
| 5.0 | `0.4078` | `0.4743` | `0.4271` | `86.0%` | `2.064` |
| 9.5 | `0.4078` | `0.4743` | `0.4271` | `86.0%` | `2.132` |

关键问题：budget 增加没有提升 learned greedy value，且 learned greedy 低于 semantic-only，说明 selector 学到的专家选择仍然不可靠。

### 2.2 Phase2 主表

| Method | R@1 | R@5 | MRR | Avg cost | Avg experts | GT filtered |
|---|---:|---:|---:|---:|---:|---:|
| B0 all feasible experts | `26.4%` | `50.0%` | `0.372` | `5.000` | `3.000` | `0.0%` |
| B1 random | `26.8%` | `56.4%` | `0.399` | `3.480` | `2.296` | `0.0%` |
| B2 fixed cascade | `26.4%` | `50.0%` | `0.372` | `5.000` | `3.000` | `0.0%` |
| B4 UCB bandit | `28.0%` | `55.6%` | `0.404` | `4.348` | `2.528` | `0.0%` |
| B-oracle | `35.2%` | `61.2%` | `0.474` | `1.730` | `1.368` | `0.0%` |
| B6 CSER | `28.0%` | `57.6%` | `0.408` | `3.184` | `2.064` | `1.6%` |

CSER 的 retrieval 指标没有明显提升，但其候选过滤统计现在可解释：

- `avg_candidates_after_filter = 244.668 / 1000`
- `candidate_reduction_rate = 75.5332%`
- `conformal_coverage = 98.0%`
- `GT_filtered_rate = 1.6%`

### 2.3 Phase2 E6 safety ablation

| Variant | R@1 | R@5 | MRR | GT filtered | Avg candidates | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| Mondrian conformal | `28.0%` | `57.6%` | `0.4078` | `1.6%` | `244.668` | `75.5332%` |
| Split conformal | `28.0%` | `57.6%` | `0.4077` | `2.8%` | `223.184` | `77.6816%` |
| Heuristic threshold | `28.0%` | `57.6%` | `0.4078` | `1.2%` | `280.772` | `71.9228%` |
| No gate | `28.0%` | `57.6%` | `0.4076` | `6.8%` | `100.000` | `90.0000%` |

这组结果说明：

1. 新 safety path 是有效的：Mondrian 把 no-gate 的 GT 误删从 `6.8%` 降到 `1.6%`。
2. Mondrian 相比 heuristic 保留候选更少，但误删更高；这是一条真实 tradeoff。
3. Split 更激进，保留候选更少，但误删高于 Mondrian。
4. R@K 基本不变，说明目前过滤影响主要体现在 safety / efficiency，不是 retrieval gain。

## 3. 建议保留的改动

### 3.1 保留 operational safety filtering

保留当前的候选构造：

```text
final_candidates(q) = semantic_top_k(q) union C(q)
```

理由：

- 它把 conformal set 从 coverage-only diagnostic 变成真实 hard constraint。
- 可以直接测量 `GT_filtered_rate`。
- 可以在 MSRVTT10K 上形成候选缩减证据。
- 对 10K 来说，候选缩减比 1K 更关键。

后续只需要调参，不应回退到旧的“不过滤，仅报告 coverage”实现。

### 3.2 保留 Phase2 safety 指标

必须保留并继续输出：

- `GT_filtered_rate`
- `avg_candidates_after_filter`
- `candidate_reduction_rate`
- `hard_filter_activation_rate`
- `conformal_coverage`
- `coverage_report`

MSRVTT10K 中只看 R@K 不够。必须同时看候选规模，否则无法证明 safety gate 对大规模检索有用。

### 3.3 保留 real-model fail-closed

显式 `--real-models` 时，如果任一专家初始化失败，应直接失败，不允许 fallback 到 mock。

理由：

- 10K 全量实验成本高，不能跑完后才发现混入 mock。
- 论文数字必须来自真实专家。
- 这符合当前 preflight 设计：`check_cser_experts.py` + `check_cser_runtime_models.py`。

### 3.4 保留 cost scope 标注

保留 `cost_kind = offline_index_expert_unit_proxy`。

当前 cost 是离线专家索引访问单位，不是端到端模型调用时延。MSRVTT10K 论文表中应明确区分：

- offline index cost proxy
- cached score rerank latency
- end-to-end expert extraction time
- query-time latency

### 3.5 保留 theorem non-vacuous 检查

Theorem 2 / Theorem 3 必须继续报告：

- `bound_RHS`
- `bound_is_non_vacuous`
- `monotonicity_violation_rate`
- `monotonicity_assumption_holds`
- `all_three_hold`

如果 RHS 为负，不能把 `bound_holds=true` 当作有效理论支持。

## 4. 必须调整的问题

### 4.1 P0: 为 MSRVTT10K 增加可复用全专家 gallery cache

当前 Slurm 输出显示 phase1、phase2、phase3 都会重新执行：

```text
[expert-extract 100/1000]
...
[expert-extract 1000/1000]
```

1K 已经重复提取三次，10K 会非常浪费。下次 MSRVTT10K 前必须把 gallery-level expert signals 落盘并复用。

建议实现：

1. 增加 `scripts/prepare_msrvtt10k_real_cache.py`，职责包括：
   - 读取 MSRVTT10K manifest / split。
   - 抽帧并运行 5 个真实专家。
   - 保存 `GallerySignals` 或等价结构。
   - 保存 query CSV 和 text embeddings。
   - 支持断点续跑。
2. 在 `cser.data.load_video_dataset` 中支持优先读取已缓存的 gallery signals。
3. phase1 / phase2 / phase3 接收同一份 cache，而不是重复 decode videos 和 expert extraction。
4. 输出 cache manifest：
   - `n_videos`
   - `n_queries`
   - `n_frames`
   - expert class names
   - checkpoint paths
   - created_at
   - feature shape / dtype
   - missing or failed videos

验收标准：

- phase1/2/3 不再重复打印 10K 次 expert extraction。
- 第二次运行能直接加载 cache。
- cache manifest 中确认 5 个 expert 都是真实类。

### 4.2 P0: 增加 candidate_top_k sweep

当前 `candidate_top_k=100` 在 1K 上得到：

- Mondrian: `GT_filtered=1.6%`, `avg_candidates=244.668`
- No gate: `GT_filtered=6.8%`, `avg_candidates=100`

10K 上 `top_k=100` 只保留 1% gallery，风险会变大。必须 sweep。

建议 10K sweep:

| candidate_top_k | 目的 |
|---:|---|
| 100 | 最激进，测极限压缩和误删风险 |
| 300 | 中等激进，接近 1K 实验的相对比例 |
| 500 | 稳健候选规模 |
| 1000 | 保守设置，检查 R@K 和 GT filtered 是否稳定 |
| 2000 | 可选，用作 safety upper bound |

每个 top-k 都要比较：

- Mondrian conformal
- Split conformal
- Heuristic threshold
- No gate

报告指标：

- `R@1`, `R@5`, `R@10`, `MRR`
- `GT_filtered_rate`
- `conformal_coverage`
- `avg_candidates_after_filter`
- `candidate_reduction_rate`
- `hard_filter_activation_rate`
- per-query candidate count percentile: p50 / p90 / p95 / p99

建议选择准则：

```text
首选配置 = GT_filtered_rate <= alpha 且 R@1 不低于 no-filter 基线 0.5pp 以上，同时 avg_candidates 尽可能小
```

如果 10K 上 `top_k=100` 误删过高，应默认升到 `top_k=500` 或 `top_k=1000`。

### 4.3 P0: 修正 CSER selector 质量问题

当前最核心的算法问题是 selector 不强：

- budget 5 learned greedy value `0.4078`
- semantic-only value `0.4271`
- oracle value `0.4743`

也就是说，learned selector 选择专家后反而低于 semantic-only。

建议分三步诊断。

#### 4.3.1 增加 per-query selection audit

为 Phase2 输出每个 query 的：

- selected expert mask
- selected expert names
- semantic-only rank / RR
- CSER rank / RR
- oracle mask under same budget
- oracle rank / RR
- delta versus semantic-only
- conformal set size
- candidate count
- gt filtered
- query text
- gt video id

保存为：

```text
e1_cser_query_audit.jsonl
```

用途：

- 找出 CSER 何时比 semantic-only 差。
- 判断是否某些专家系统性伤害排序。
- 为 MSRVTT10K debug 提供可抽样证据。

#### 4.3.2 增加 expert mask distribution

输出：

- 每个 mask 被选择的次数。
- 每个 expert 被选择的频率。
- 每个 mask 的平均 RR / R@1。
- 每个 mask 相对 semantic-only 的平均 delta。

如果大多数 query 都选择同一个低效 mask，说明 SVN policy collapse。

#### 4.3.3 加入 semantic fallback rule

在 learned selector 未稳定前，建议加入保守 fallback：

```text
if predicted best marginal gain <= tau:
    use semantic-only
```

或者：

```text
if selected mask historically has negative mean delta on validation:
    fallback to semantic-only
```

这不是最终算法创新点，但对 10K 全量实验很重要，因为它能避免 learned selector 系统性伤害检索。

建议 sweep:

| fallback | tau |
|---|---:|
| none | N/A |
| predicted marginal threshold | 0.00 |
| predicted marginal threshold | 0.01 |
| validation negative-mask blocklist | N/A |

### 4.4 P1: 处理 monotonicity violation

当前 monotonicity violation `17.425%`。这会削弱理论叙事。

建议不要在下次实验前强行修正理论，而是同时做两个版本：

1. **Raw objective**: 保持当前真实 value lattice，用于诚实报告。
2. **Monotone envelope objective**: 对 value matrix 做单调包络：

```text
V_mono(S) = max_{T subseteq S} V(T)
```

用途：

- 训练 SVN 时不再奖励会降低 value 的专家添加。
- 检查 monotone objective 是否提高 greedy selector 稳定性。
- 理论表可以单独报告 raw vs monotone-envelope。

需要报告：

- raw monotonicity violation
- monotone-envelope violation
- raw CSER R@K
- envelope-trained CSER R@K
- selector mask distribution
- oracle gap

### 4.5 P1: `face_id` 专家在文本查询协议下需要降权或禁用

当前文本 query 没有 reference face embedding，`face_id` 难以贡献真实信息。MSRVTT10K 如果仍然是 text-to-video，建议：

1. 默认禁用 `face_id`，跑一个 `without_face_id` 版本。
2. 或保留但设置 `available=False`，让 selector 不选择它。
3. 单独设计 face-reference 子任务时再启用。

建议实验矩阵：

| Expert roster | 说明 |
|---|---|
| all_optional | 当前完整 4 optional experts |
| no_face_id | 文本查询主协议建议版本 |
| semantic_highlight_scene | 更轻量版本 |

验收标准：

- 如果 `no_face_id` 不降低 R@K 且降低 cost，应优先作为 MSRVTT10K 主表版本。

### 4.6 P1: scene expert 需要重新校准

1K 结果中 scene 贡献弱。10K 前建议输出 scene 诊断：

- query 中有 scene cue 的比例。
- gallery scene label 分布。
- scene cue 与 GT scene label match rate。
- scene expert 单独加入时的 delta RR。
- scene expert 被 selector 选择时的实际收益。

如果 scene cue / classifier label vocabulary 不一致，应先做 mapping 修正，而不是扩大到 10K 后再分析。

### 4.7 P1: 端到端 latency 必须单独定义

当前 E7 应继续标注为 cached-score rerank timing，不应作为端到端加速证据。

MSRVTT10K 建议分三类时间：

1. **Offline build time**
   - decode frames
   - MobileCLIP
   - MomentDETR
   - SCRFD
   - ArcFace
   - MobileNetV3
2. **Query-time rerank time**
   - semantic score
   - candidate top-k
   - conformal mask
   - optional expert score fusion
3. **End-to-end amortized time**
   - offline build time amortized over query count
   - query-time time

如果 InsightFace 继续使用 CPU provider，必须在报告中写明：

```text
InsightFace ran with CPUExecutionProvider; GPU end-to-end latency is not validated.
```

### 4.8 P1: 明确 10K 协议和 split

下次 MSRVTT10K 必须在 manifest 中固定：

- gallery size: 10K
- query count
- captions per video
- train / calibration / test split
- 是否与官方 test set 重叠
- router train 是否使用最终 test query
- conformal calibration 是否独立于 final test

建议至少提供两个协议：

| Protocol | 目的 |
|---|---|
| dev protocol | 允许在 10K 上调 top-k / gate / selector |
| final protocol | 固定参数后只跑一次最终表 |

不要在同一批 final test query 上反复调 `candidate_top_k` 后再报告最优结果为最终性能。

## 5. MSRVTT10K 推荐实验流程

### Step 0: Preflight

必须先跑：

```bash
python -B scripts/check_cser_experts.py --out reports/setup/cser_expert_status.json
python -B scripts/check_cser_runtime_models.py
```

检查：

- 5 个 expert class 都来自 `tasks.real_models`
- 没有 mock class
- MobileCLIP / MomentDETR checkpoint 路径正确
- InsightFace provider 状态记录清楚

### Step 1: 10K cache build

新增或扩展脚本，目标输出：

```text
data/msrvtt_real_10k/
  manifest.json
  msrvtt10k_cache.npz
  msrvtt10k_queries.csv
  msrvtt10k.text_embs.npy
  cser_gallery_signals.npz
  frame_features/
```

`manifest.json` 中必须包含：

- source paths
- split file paths
- model checkpoint paths
- expert class names
- n_videos
- n_queries
- captions_per_video
- n_frames
- failed video ids
- created_at

### Step 2: 10K Phase1

先只跑 Phase1，目的不是发主表，而是确认 value lattice 是否稳定：

- submodularity violation
- monotonicity violation
- gamma
- greedy vs oracle
- semantic-only vs learned greedy

如果 learned greedy 仍低于 semantic-only，不应直接进入最终主表，而应先执行 selector audit。

### Step 3: 10K Phase2 safety sweep

建议实验矩阵：

| candidate_top_k | gate | expert roster | selector |
|---:|---|---|---|
| 100 | no gate | all | learned |
| 100 | Mondrian | all | learned |
| 300 | Mondrian | all | learned |
| 500 | Mondrian | all | learned |
| 1000 | Mondrian | all | learned |
| 500 | Split | all | learned |
| 500 | heuristic | all | learned |
| 500 | Mondrian | no_face_id | learned |
| 500 | Mondrian | all | semantic fallback |

主指标：

- `R@1`, `R@5`, `R@10`, `MRR`
- `GT_filtered_rate`
- `avg_candidates_after_filter`
- `candidate_reduction_rate`
- `avg_cost`
- `avg_experts_called`
- p50 / p90 / p95 / p99 candidate count

### Step 4: Select configuration

建议配置选择规则：

1. `GT_filtered_rate <= alpha`，优先满足 safety。
2. `R@1` 不低于 no-filter / semantic-only 超过 `0.5pp`。
3. 在满足 1 和 2 的配置中，选择 `avg_candidates_after_filter` 最小者。
4. 如果 learned selector 低于 semantic-only，则主表应报告 semantic fallback 或 no_face_id 版本，learned selector 放入 ablation。

### Step 5: Final protocol run

固定参数后，再跑最终表。最终表不要再选择最优 top-k。

最终报告至少包含：

- main retrieval table
- safety table
- candidate reduction table
- selector audit summary
- expert contribution table
- latency scope table
- theorem diagnostics table

## 6. 建议代码改动清单

### 6.1 新增缓存复用能力

建议文件：

- `scripts/prepare_msrvtt_real_10k.py`
- `scripts/check_cser_gallery_cache.py`
- `cser/gallery_cache.py`

核心接口：

```python
save_gallery_signals(gallery, path, metadata)
load_gallery_signals(path) -> GallerySignals
```

`load_video_dataset` 增加参数：

```text
--gallery-cache /path/to/cser_gallery_signals.npz
```

### 6.2 Phase2 增加 top-k sweep mode

建议新增参数：

```text
--candidate-top-k-list 100 300 500 1000
```

输出：

```text
e6_candidate_topk_sweep.json
```

### 6.3 Phase2 增加 query audit

建议新增参数：

```text
--write-query-audit
```

输出：

```text
e1_cser_query_audit.jsonl
selector_mask_distribution.json
expert_delta_summary.json
```

### 6.4 增加 no_face_id roster

建议新增参数：

```text
--disabled-experts face_id
```

或更明确：

```text
--expert-roster all
--expert-roster no_face_id
--expert-roster semantic_highlight_scene
```

### 6.5 增加 monotone objective variant

建议新增参数：

```text
--value-transform raw
--value-transform monotone_envelope
```

输出时必须区分：

- `metric`
- `value_transform`
- `monotonicity_violation_rate`

## 7. 下次实验前的验收标准

在提交 MSRVTT10K Slurm 前，建议满足：

- `check_cser_runtime_models.py` 通过，且无 mock expert。
- 10K gallery cache 可加载，不重复提取专家。
- `candidate_top_k` sweep 可在小样本或 1K 上跑通。
- Phase2 可输出 query audit。
- `GT_filtered_rate`、candidate count percentile 都写入结果。
- `--real-models` fail-closed 行为保留。
- 文档明确 latency scope。
- final protocol 的 split 和 caption policy 固定。

## 8. 当前结论可用于论文的程度

可以谨慎使用：

> Conformal safety filtering substantially reduces false elimination compared
> with aggressive no-gate top-k filtering, while preserving the observed R@K in
> the 1K diagnostic setting.

不建议使用：

> CSER improves retrieval accuracy over all baselines.

不建议使用：

> CSER provides a validated non-vacuous combined theorem guarantee on the real
> MSR-VTT run.

不建议使用：

> CSER achieves real end-to-end GPU speedup.

## 9. 推荐优先级

| Priority | Work item | Why |
|---|---|---|
| P0 | 10K gallery expert cache | 否则全量实验成本过高且重复 |
| P0 | candidate_top_k sweep | 10K 上 top-100 风险未知 |
| P0 | query audit / selector diagnosis | learned selector 仍弱于 semantic-only |
| P1 | no_face_id roster | 文本查询下 face_id 不可靠 |
| P1 | monotone envelope variant | 缓解 monotonicity violation 与理论叙事冲突 |
| P1 | latency scope split | 防止 cached rerank 被误写为 end-to-end |
| P2 | scene expert mapping audit | 修复弱 scene 贡献 |
| P2 | final protocol lock | 防止在 final test 上调参 |
