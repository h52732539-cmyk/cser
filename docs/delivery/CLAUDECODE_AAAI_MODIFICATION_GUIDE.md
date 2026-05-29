# Claude Code 修改指南：将 LiteVTR++ 改造成 AAAI 主会可投版本

> 目标读者：Claude Code / 代码执行代理 / 项目实现者  
> 当前任务：在现有 LiteVTR++ 工程基础上，补齐 **Calibrated Query-Intent Planning** 的核心方法、实验、baseline 和论文表述，使项目从“强工程系统”升级为“可投稿 AAAI 主会的 AI 方法论文”。  
> 生成日期：2026-05-09  
> 参考材料：`PROJECT_SUMMARY_FOR_REVIEW.md` 与 `粘贴的文本 (1)(5).txt`

---

## 0. 一句话总目标

把当前的：

> rule-based QueryParser + synthetic metadata upper bound + 工程加速系统

改造成：

> **冻结多模态专家约束下的预算化视频检索规划框架**，核心方法是一个 **2-head calibrated QIN**：
>
> 1. **Route value head**：预测每条候选 retrieval route 的 utility；
> 2. **Filter safety head**：预测每个 metadata axis 做 hard filter 时 GT 是否能存活，并用 calibration 控制误删风险。

注意：不要把项目写成“手机相册加速系统”。论文主线必须是：

> **Black-box budgeted video retrieval with frozen multimodal experts**

---

## 1. 必须遵守的核心约束

### 1.1 Frozen expert constraint

所有已有视觉 / 多模态模型都必须视为 frozen black-box experts：

- 不允许 fine-tuning；
- 不允许 quantization；
- 不允许 architecture change；
- 不允许 distillation 成替代模型；
- 只允许决定：
  - 调用哪些模型；
  - 调用哪些帧；
  - 使用哪些 retrieval axes；
  - 哪些 axes 做 hard filter；
  - 哪些 axes 做 soft rerank；
  - candidate size / route budget / rerank budget。

### 1.2 不能再主打 synthetic 69.5% R@1

当前项目中的 Phase 3 `+ Metadata filter = 69.5% R@1` 使用的是 **synthetic metadata**，且 query constraints matched to GT。这个结果只能作为：

```text
Oracle / controlled upper bound
```

不能出现在 abstract、main claim、main table 的第一主结论中。

正确处理方式：

```text
Main results:
  realistic noisy metadata / real metadata / controlled non-oracle metadata

Appendix or controlled study:
  perfect synthetic metadata upper bound = 69.5% R@1
```

### 1.3 不要实现 4-head QIN

不要实现以下过设计版本：

```text
axis relevance head
filter safety head
route value head
stop/escalate head
```

请实现简洁版本：

```text
2-head C-QIN / CalQIN:
  Head 1: route value head
  Head 2: filter safety head
```

axis relevance、stop/escalate、fusion alpha 都应该从 route value 或 route definition 中隐式得到，不要额外堆 head。

### 1.4 理论部分不要凑平凡 theorem

不要把下面这个作为核心理论贡献：

```text
If |Û - U| <= ε, then regret <= 2ε.
```

这太平凡。最多作为 appendix sanity lemma。主文理论只保留：

```text
calibrated hard filtering / empirical risk control / conformal-style calibration
```

并且必须诚实写明：任何 finite-sample validity 需要 calibration/test exchangeability 或近似同分布假设。

---

## 2. 推荐命名

论文和代码可以使用以下名字之一：

```text
C-QIN: Calibrated Query-Intent Network
CalQIN: Calibrated Query-Intent Network
CQP: Calibrated Query Planner
```

本文档默认使用 **C-QIN**。

不推荐使用：

```text
SAFE-QIN
```

原因：SAFE 不够直接，容易显得是 marketing name。AAAI 论文更适合强调 calibrated / black-box / budgeted / frozen experts。

---

## 3. 代码改造总览

请优先添加以下模块。若现有 repo 结构不同，请适配到已有目录，但保持功能边界一致。

```text
litevtr/
  routing/
    __init__.py
    route_schema.py          # Route dataclass / schema validation
    route_bank.py            # 约 30 条代表性 candidate routes
    route_executor.py        # 执行一条 route，返回 rank/cost/filter stats
    route_bank_builder.py    # 离线枚举 route bank，生成 counterfactual labels
    qin_model.py             # 2-head C-QIN model
    train_qin.py             # 训练 route value + safety heads
    calibrated_planner.py    # inference-time planner with safety threshold
    calibrate_safety.py      # calibration threshold selection
    baselines.py             # semantic/rule/QPP/random/oracle/always-hard/cascade

  metadata/
    noisy_metadata.py        # realistic noisy metadata injection
    metadata_schema.py       # time/geo/motion/device schema

  eval/
    metrics.py               # R@K, MeanR, MRR, GT filtered rate, cost metrics
    eval_planner.py          # planner evaluation entrypoint
    eval_routes.py           # route bank/oracle evaluation
    report_tables.py         # main table / ablation table export

configs/
  route_bank_30.yaml
  qin_train.yaml
  metadata_noise_msrvtt.yaml
  eval_main.yaml

scripts/
  build_route_bank.sh
  train_cqin.sh
  calibrate_cqin.sh
  eval_cqin.sh
  make_aaai_tables.sh

reports/
  aaai_main/
    main_results.csv
    ablation_results.csv
    calibration_results.csv
    oracle_gap.csv
    route_bank_summary.csv
    metadata_noise_sweep.csv
```

---

## 4. Phase A：先做 repo audit 与结果冻结

### 4.1 Claude Code 第一步必须做什么

先不要直接改模型。请先扫描 repo，生成现有模块映射：

```text
Current module map:
  QueryParser: <path>
  MetaFilter: <path>
  OfflineIndex: <path>
  QPP Planner: <path>
  NNN/QAMP rerank: <path>
  col-softmax: <path>
  UnifiedScheduler: <path>
  CrossTaskCache: <path>
  evaluation scripts: <path>
  dataset configs: <path>
```

如果现有代码没有这些模块名，就通过功能定位。

### 4.2 冻结现有 baseline

在改造前，必须保存当前结果作为 baseline snapshot：

```text
reports/baseline_snapshot/
  msrvtt_phase2_semantic.json
  msrvtt_phase3_synthetic_upper_bound.json
  retrieval_ablation_existing.csv
  sampling_ablation_existing.csv
  run_config.yaml
  git_commit.txt
```

必须明确标记：

```text
phase3_synthetic_upper_bound = true
metadata_setting = perfect_synthetic_matched_to_gt
```

---

## 5. Phase B：定义有限 route bank，避免组合爆炸

### 5.1 为什么必须限制 route bank

不要枚举完整笛卡尔积：

```text
5 axes × hard/soft/off = 3^5 = 243
再乘 candidate_topm、threshold、budget、model policy，很快上万条 route
```

这会导致离线评估成本过高，也会让方法显得混乱。

请实现一个固定 route vocabulary：

```text
K_route ≈ 30
```

原则：

1. 覆盖常见 route family；
2. 每条 route 可解释；
3. 能产生 oracle route 上界；
4. 能支撑 C-QIN 学习；
5. 能做 budget curve。

### 5.2 Route schema

在 `litevtr/routing/route_schema.py` 中定义：

```python
from dataclasses import dataclass
from typing import Literal

Axis = Literal["semantic", "time", "geo", "motion", "device"]
RerankMode = Literal["none", "qamp", "nnn_qamp", "col_softmax_post_filter"]
BudgetTier = Literal["low", "medium", "high", "full"]

@dataclass(frozen=True)
class RetrievalRoute:
    route_id: str
    description: str
    hard_axes: tuple[Axis, ...]
    soft_axes: tuple[Axis, ...]
    candidate_topm: int
    rerank_mode: RerankMode
    budget_tier: BudgetTier
    use_offline_index: bool = True
    allow_image_model_calls: bool = False
    allow_dense_refinement: bool = False
```

Validation rules:

```text
- semantic 不允许出现在 hard_axes；semantic 是基础检索轴。
- hard_axes 与 soft_axes 不允许重叠。
- hard_axes 只能来自 time/geo/motion/device。
- candidate_topm 必须在 {100, 300, 500, 1000}。
- low budget route 不允许 allow_dense_refinement=True。
```

### 5.3 推荐 30 条 route

在 `configs/route_bank_30.yaml` 中实现。可以先用以下 route family：

```yaml
routes:
  - route_id: R00_semantic_only_top500
    description: Semantic OfflineIndex only, topM=500
    hard_axes: []
    soft_axes: []
    candidate_topm: 500
    rerank_mode: nnn_qamp
    budget_tier: low
    allow_image_model_calls: false
    allow_dense_refinement: false

  - route_id: R01_semantic_only_top300
    description: Semantic OfflineIndex only, topM=300
    hard_axes: []
    soft_axes: []
    candidate_topm: 300
    rerank_mode: nnn_qamp
    budget_tier: low

  - route_id: R02_time_hard_top500
    hard_axes: [time]
    soft_axes: []
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R03_geo_hard_top500
    hard_axes: [geo]
    soft_axes: []
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R04_motion_hard_top500
    hard_axes: [motion]
    soft_axes: []
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R05_device_hard_top500
    hard_axes: [device]
    soft_axes: []
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R06_time_geo_hard_top500
    hard_axes: [time, geo]
    soft_axes: []
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R07_time_motion_hard_top500
    hard_axes: [time, motion]
    soft_axes: []
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R08_geo_motion_hard_top500
    hard_axes: [geo, motion]
    soft_axes: []
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R09_time_geo_motion_hard_top500
    hard_axes: [time, geo, motion]
    soft_axes: []
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: medium

  - route_id: R10_time_hard_motion_soft_top500
    hard_axes: [time]
    soft_axes: [motion]
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: medium

  - route_id: R11_geo_hard_motion_soft_top500
    hard_axes: [geo]
    soft_axes: [motion]
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: medium

  - route_id: R12_time_hard_geo_soft_top500
    hard_axes: [time]
    soft_axes: [geo]
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: medium

  - route_id: R13_geo_hard_time_soft_top500
    hard_axes: [geo]
    soft_axes: [time]
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: medium

  - route_id: R14_soft_time_geo_motion_top500
    hard_axes: []
    soft_axes: [time, geo, motion]
    candidate_topm: 500
    rerank_mode: nnn_qamp
    budget_tier: medium

  - route_id: R15_semantic_top100_no_rerank
    hard_axes: []
    soft_axes: []
    candidate_topm: 100
    rerank_mode: none
    budget_tier: low

  - route_id: R16_semantic_top1000_nnn_qamp
    hard_axes: []
    soft_axes: []
    candidate_topm: 1000
    rerank_mode: nnn_qamp
    budget_tier: medium

  - route_id: R17_time_hard_top300
    hard_axes: [time]
    soft_axes: []
    candidate_topm: 300
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R18_geo_hard_top300
    hard_axes: [geo]
    soft_axes: []
    candidate_topm: 300
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R19_time_geo_hard_top300
    hard_axes: [time, geo]
    soft_axes: []
    candidate_topm: 300
    rerank_mode: col_softmax_post_filter
    budget_tier: low

  - route_id: R20_time_hard_top1000
    hard_axes: [time]
    soft_axes: []
    candidate_topm: 1000
    rerank_mode: col_softmax_post_filter
    budget_tier: medium

  - route_id: R21_geo_hard_top1000
    hard_axes: [geo]
    soft_axes: []
    candidate_topm: 1000
    rerank_mode: col_softmax_post_filter
    budget_tier: medium

  - route_id: R22_time_geo_hard_top1000
    hard_axes: [time, geo]
    soft_axes: []
    candidate_topm: 1000
    rerank_mode: col_softmax_post_filter
    budget_tier: medium

  - route_id: R23_motion_soft_top500
    hard_axes: []
    soft_axes: [motion]
    candidate_topm: 500
    rerank_mode: nnn_qamp
    budget_tier: low

  - route_id: R24_device_soft_top500
    hard_axes: []
    soft_axes: [device]
    candidate_topm: 500
    rerank_mode: nnn_qamp
    budget_tier: low

  - route_id: R25_time_soft_top500
    hard_axes: []
    soft_axes: [time]
    candidate_topm: 500
    rerank_mode: nnn_qamp
    budget_tier: low

  - route_id: R26_geo_soft_top500
    hard_axes: []
    soft_axes: [geo]
    candidate_topm: 500
    rerank_mode: nnn_qamp
    budget_tier: low

  - route_id: R27_time_geo_soft_top500
    hard_axes: []
    soft_axes: [time, geo]
    candidate_topm: 500
    rerank_mode: nnn_qamp
    budget_tier: medium

  - route_id: R28_time_geo_hard_dense_refine
    hard_axes: [time, geo]
    soft_axes: []
    candidate_topm: 500
    rerank_mode: col_softmax_post_filter
    budget_tier: high
    allow_image_model_calls: true
    allow_dense_refinement: true

  - route_id: R29_full_budget_all_soft_dense_refine
    hard_axes: []
    soft_axes: [time, geo, motion, device]
    candidate_topm: 1000
    rerank_mode: nnn_qamp
    budget_tier: full
    allow_image_model_calls: true
    allow_dense_refinement: true
```

如果实际系统没有 device/motion metadata，可暂时保留 schema，但禁用对应 route 或让它们在 metadata_missing 时自动退化。

---

## 6. Phase C：实现 counterfactual route bank builder

### 6.1 目标

对每个 query，离线执行所有 candidate routes，并记录每条 route 的效果：

```text
query_id
route_id
gt_video_id
rank
recall@1 / recall@5 / recall@10
mrr
gt_filtered
candidate_count_after_filter
candidate_topm
latency_ms
model_calls
npu_active_ms_proxy
energy_proxy
utility
```

输出文件：

```text
reports/route_bank/msrvtt_noisy_route_bank.parquet
reports/route_bank/msrvtt_noisy_route_bank_summary.csv
```

### 6.2 Utility 定义

先实现一个简单、可解释、可调的 utility：

```python
def retrieval_gain(rank: int) -> float:
    if rank <= 0:  # not found / filtered out
        return 0.0
    return 1.0 / rank  # MRR-style gain

utility = (
    gain_weight * retrieval_gain(rank)
    + hit1_weight * int(rank == 1)
    + hit5_weight * int(1 <= rank <= 5)
    - cost_weight * normalized_cost
    - filter_penalty * int(gt_filtered)
)
```

推荐默认值：

```yaml
gain_weight: 1.0
hit1_weight: 0.5
hit5_weight: 0.1
cost_weight: 0.05
filter_penalty: 2.0
```

必须支持通过 config 调整。

### 6.3 Oracle route

对每个 query：

```python
oracle_route = argmax_route utility(query, route)
```

记录：

```text
oracle_route_id
oracle_utility
oracle_rank
oracle_cost
```

后续用于：

1. 训练 route value head；
2. 计算 QIN oracle gap；
3. 作为 AAAI baseline。

### 6.4 Hard-filter survival label

对每个 axis 构造 safety label：

```python
survival_label[axis] = 1 if gt_video survives hard filtering by this axis else 0
```

axis 包括：

```text
time
geo
motion
device
```

注意：semantic 不是 hard-filter axis，不需要 safety label。

输出：

```text
query_id
time_survive
geo_survive
motion_survive
device_survive
```

---

## 7. Phase D：构造 realistic noisy metadata 设置

### 7.1 目标

用 realistic noisy metadata 替代 perfect synthetic metadata 作为 controlled main experiment。

当前 perfect synthetic metadata 的问题：

```text
random GPS/time assigned to real videos
query constraints matched to GT
```

这会产生 oracle-like filter signal，不能作为主结果。

### 7.2 在 MSR-VTT 上实现 metadata noise injection

文件：`litevtr/metadata/noisy_metadata.py`

请支持以下 noise 类型：

```yaml
metadata_noise:
  time:
    enabled: true
    shift_days_std: 7
    missing_prob: 0.2
    ambiguous_cluster_prob: 0.3

  geo:
    enabled: true
    jitter_km_std: 20
    wrong_region_prob: 0.1
    missing_prob: 0.3
    ambiguous_cluster_prob: 0.3

  motion:
    enabled: true
    label_flip_prob: 0.15
    missing_prob: 0.2

  device:
    enabled: true
    label_flip_prob: 0.05
    missing_prob: 0.1
```

### 7.3 必须包含 4 个 metadata 设置

实验至少输出四种设置：

```text
S0_semantic_only:
  no metadata, Phase 2 baseline

S1_perfect_synthetic_upper_bound:
  existing Phase 3, only upper bound

S2_realistic_noisy_synthetic:
  main controlled metadata experiment

S3_real_or_external_metadata:
  optional; YFCC100M video subset or RealAlbum if available
```

### 7.4 不要把 Ego4D 当真实 GPS 数据集

Ego4D 可以用于：

```text
long video
NLQ
IMU / motion
temporal localization
```

不要写：

```text
Ego4D provides real GPS metadata
```

除非实际确认某个 split 有 GPS 字段，并在文档中列出字段来源。

---

## 8. Phase E：实现 2-head C-QIN

### 8.1 输入特征

C-QIN 输入建议：

```text
frozen_CLIP_text_embedding: 512D
QPP statistics: 6D
keyword / parser weak indicators: 5D
metadata availability vector: 4D
budget vector: 3D or 4D
```

合计约：

```text
527D 左右，取决于 budget encoding
```

QPP statistics 可包括：

```text
top1_score
top2_score
margin_top1_top2
entropy_topk
score_std_topk
rank_concentration
```

keyword indicators 可包括：

```text
has_time_hint
has_geo_hint
has_motion_hint
has_device_hint
has_event_hint
```

metadata availability vector：

```text
time_available
geo_available
motion_available
device_available
```

budget vector：

```text
budget_low
budget_medium
budget_high
budget_full
```

### 8.2 Model architecture

文件：`litevtr/routing/qin_model.py`

推荐结构：

```python
class CalibratedQIN(nn.Module):
    def __init__(self, input_dim: int, num_routes: int, num_safety_axes: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.route_value_head = nn.Linear(64, num_routes)
        self.filter_safety_head = nn.Linear(64, num_safety_axes)

    def forward(self, x):
        h = self.encoder(x)
        route_values = self.route_value_head(h)
        safety_logits = self.filter_safety_head(h)
        return {
            "route_values": route_values,
            "safety_logits": safety_logits,
            "safety_probs": torch.sigmoid(safety_logits),
        }
```

参数量应保持 < 100K。

### 8.3 Loss function

训练目标由三部分组成。

#### Loss 1：route utility regression

```python
loss_value = huber_loss(predicted_route_values, normalized_route_utilities)
```

#### Loss 2：oracle route classification

```python
oracle_route_id = argmax(route_utilities)
loss_route_ce = cross_entropy(predicted_route_values, oracle_route_id)
```

也可使用 soft labels：

```python
soft_target = softmax(route_utilities / temperature)
loss_route_kl = KLDivLoss(log_softmax(predicted_route_values), soft_target)
```

#### Loss 3：filter safety prediction

```python
loss_safety = BCEWithLogitsLoss(safety_logits, survival_labels)
```

总 loss：

```python
loss = loss_value + alpha * loss_route_ce + beta * loss_safety
```

默认：

```yaml
alpha: 1.0
beta: 1.0
```

### 8.4 训练 split

必须严格拆分：

```text
train split:
  fit C-QIN weights

calibration split:
  select safety thresholds tau_axis

test split:
  final reporting only
```

禁止在 test set 上选择 calibration threshold。

---

## 9. Phase F：实现 calibrated hard filtering

### 9.1 目标

控制 hard filter 把 GT 错删的风险。

需要报告：

```text
GT filtered rate ↓
accepted hard-filter coverage ↑
R@1 / R@5
cost / latency
```

### 9.2 Practical calibration algorithm

不要过度承诺严格 distribution-free guarantee。先实现一个可靠、可解释的 calibration 过程。

对每个 axis `a`：

```python
inputs on calibration set:
  safety_score_i = sigmoid(safety_logit_i[a])
  failure_i = 1 - survival_label_i[a]
```

选择 threshold `tau_a`：

```python
for tau in sorted(unique_scores):
    accepted = safety_score >= tau
    n = accepted.sum()
    k = failure[accepted].sum()
    empirical_failure_rate = k / n
    ucb_failure_rate = binomial_clopper_pearson_upper(k, n, alpha=0.05)

choose the lowest tau such that:
    n >= min_accept
    ucb_failure_rate <= delta
```

推荐默认：

```yaml
delta: 0.05
alpha: 0.05
min_accept: 30
```

如果某个 axis 没有足够 calibration 支持：

```text
disable hard filter for that axis by default
```

### 9.3 Inference-time planner

文件：`litevtr/routing/calibrated_planner.py`

推理流程：

```python
features = build_features(query)
out = cqin(features)
route_values = out["route_values"]
safety_probs = out["safety_probs"]

valid_routes = []
for route in route_bank:
    safe = True
    for axis in route.hard_axes:
        if safety_probs[axis] < tau_axis[axis]:
            safe = False
            break
    if safe:
        valid_routes.append(route)

if not valid_routes:
    route = fallback_route  # semantic only or soft-only route
else:
    route = argmax_valid_route(route_values)

return execute(route)
```

Fallback route 建议：

```text
R00_semantic_only_top500
```

### 9.4 Calibration result table

必须输出：

```text
axis
tau_axis
calib_accept_rate
calib_failure_rate
test_accept_rate
test_failure_rate
R@1_with_axis
```

输出路径：

```text
reports/aaai_main/calibration_results.csv
```

---

## 10. Phase G：实现 baselines

必须实现以下 baselines。不要跳过 Oracle route 和 Always-hard-filter-all。

### 10.1 Baseline list

```text
B0 Semantic-only OfflineIndex
B1 Existing rule-based QueryParser + MetaFilter
B2 QPP-only router
B3 Random route
B4 Oracle route
B5 Always-hard-filter-all detected axes
B6 C-QIN without calibration
B7 C-QIN with calibration
B8 Cascade / cost-sensitive baseline
B9 LLM parser/router baseline, optional but recommended
```

### 10.2 Baseline definitions

#### B0 Semantic-only OfflineIndex

使用现有 Phase 2：

```text
OfflineIndex + NNN/QAMP + no metadata filtering
```

#### B1 Rule-based QueryParser

使用现有 Phase 3 parser，但必须在 noisy metadata 上评估。

#### B2 QPP-only router

根据 QPP margin 决定 route：

```text
EASY:
  semantic-only low budget
MEDIUM:
  semantic + soft metadata rerank or conservative hard filter
HARD:
  high budget / dense refine / larger topM
```

#### B3 Random route

从 route bank 中均匀随机选择。固定 random seed，重复 5 次报告均值和标准差。

#### B4 Oracle route

对每个 query 从 route bank 中选择 utility 最大的 route。只作为 upper bound，不作为实际可用系统。

#### B5 Always-hard-filter-all detected axes

只要 query parser 检测到 axis，就 hard filter。用于证明 naive hard filtering 风险高。

#### B6 C-QIN without calibration

直接选择 route value 最高的 route，不做 safety threshold gating。

#### B7 C-QIN with calibration

最终主方法。

#### B8 Cascade / cost-sensitive baseline

实现一个简单 cascade ranking baseline：

```text
Stage 1: semantic topM
Stage 2: if QPP margin low, apply metadata filter/rerank
Stage 3: if still low confidence, use high-budget route
```

目的不是打败所有 IR 方法，而是回应“这是不是传统 cascade retrieval”的审稿质疑。

#### B9 LLM parser/router baseline

可选。如果时间允许，用一个开源 LLM 或 prompt parser 抽取 time/geo/motion/device intent，然后走相同 MetaFilter。必须报告 latency 或将其标为 offline/oracle parser。

---

## 11. Phase H：核心指标

### 11.1 Retrieval metrics

必须报告：

```text
R@1
R@5
R@10
MeanR
MRR
```

### 11.2 Safety metrics

必须报告：

```text
GT filtered rate
hard-filter activation rate
axis-wise false hard-filter rate
fallback rate
```

定义：

```python
gt_filtered = int(gt_video_id not in candidates_after_hard_filter)
GT_filtered_rate = mean(gt_filtered)
```

### 11.3 Efficiency metrics

必须报告：

```text
ms/query
model_calls/query
candidate_count_after_filter
NPU_active_ms_proxy or measured NPU_active_ms
energy_proxy or measured energy
```

如果没有真机功耗，只能写：

```text
power proxy / model-call proxy
```

不要写成 measured energy。

### 11.4 Oracle gap

必须报告：

```python
oracle_gap_utility = oracle_utility - method_utility
oracle_gap_r1 = oracle_r1 - method_r1
```

这能证明 C-QIN 是否接近 route bank 上限。

---

## 12. Phase I：主要实验表格

### 12.1 Main table

输出路径：

```text
reports/aaai_main/main_results.csv
```

表头：

```text
method
metadata_setting
R@1
R@5
R@10
MeanR
MRR
GT_filtered_rate
hard_filter_activation_rate
fallback_rate
ms_per_query
model_calls_per_query
candidate_count
oracle_gap_R@1
```

主表必须至少包含：

```text
Semantic-only
Rule parser
QPP-only
Always-hard-filter-all
C-QIN w/o calibration
C-QIN calibrated
Oracle route
```

### 12.2 Calibration table

```text
axis
tau
calib_accept_rate
calib_failure_rate_ucb
test_accept_rate
test_failure_rate
```

### 12.3 Ablation table

必须围绕 novelty，而不是工程零件。

```text
Full C-QIN calibrated
w/o calibration
w/o counterfactual route bank: train only on parser labels
w/o route value head: safety-only heuristic
w/o safety head: route value only
w/o QPP features
w/o keyword indicators
w/o budget vector
soft metadata fusion instead of hard filter
```

### 12.4 Metadata noise sweep

至少扫：

```text
geo missing prob: 0.0 / 0.1 / 0.3 / 0.5
time shift std: 0 / 3 / 7 / 14 days
wrong geo prob: 0.0 / 0.05 / 0.1 / 0.2
```

输出：

```text
R@1 vs noise
GT_filtered_rate vs noise
hard_filter_activation_rate vs noise
```

### 12.5 Upper bound table

把已有 69.5% 放这里：

```text
Controlled upper bound: perfect synthetic metadata
```

不要和 main results 混在一起。

---

## 13. Phase J：测试与验收标准

### 13.1 Unit tests

新增测试：

```text
tests/routing/test_route_schema.py
  - invalid hard_axes rejected
  - hard_axes and soft_axes cannot overlap
  - candidate_topm validation

 tests/routing/test_route_bank.py
  - route bank size between 25 and 40
  - all route IDs unique
  - route bank has semantic-only fallback
  - route bank has oracle-compatible utility fields

 tests/routing/test_route_executor.py
  - gt_filtered is correct
  - candidate count after hard filter is correct
  - post-filter col-softmax only applied after filtering

 tests/routing/test_qin_model.py
  - output shape: route_values = [B, num_routes]
  - output shape: safety_logits = [B, 4]
  - parameter count < 100K

 tests/routing/test_calibration.py
  - threshold selection handles zero accepted samples
  - threshold selection disables unsupported axis
  - calibrated planner falls back when no route is safe

 tests/eval/test_metrics.py
  - R@K correct
  - MeanR correct
  - GT filtered rate correct
```

### 13.2 Integration tests

```text
- Run route bank builder on a tiny synthetic dataset.
- Train C-QIN for 2 epochs on tiny data.
- Calibrate thresholds.
- Evaluate calibrated planner.
- Ensure output CSVs are created.
```

### 13.3 Minimum acceptance criteria

代码层面：

```text
- route bank can be built end-to-end
- C-QIN can train without crashing
- calibration split is separate from train/test
- main result table exports successfully
- all new tests pass
```

论文/实验层面：

```text
- C-QIN calibrated must be compared with rule parser and always-hard-filter-all
- GT filtered rate must be reported
- Oracle route must be reported
- 69.5% perfect synthetic must be marked as upper bound
- Any energy result must be labeled measured or proxy
```

不要硬编码“必须提升多少点”。如果结果不好，也要如实输出，因为这决定是否继续 AAAI 或转投 MLSys/ACM MM。

---

## 14. 论文文本需要同步修改

### 14.1 Abstract 主线

不要写：

```text
We build an efficient mobile gallery retrieval system.
```

建议写：

```text
We study black-box budgeted video retrieval, where multiple frozen multimodal experts must be routed under strict on-device compute budgets. We propose C-QIN, a calibrated query-intent planner that learns route values from counterfactual retrieval plans and activates hard metadata filters only when their calibrated safety scores indicate low false-elimination risk.
```

### 14.2 Contributions

推荐写 3 个 contribution：

```text
1. We formulate black-box budgeted video retrieval with frozen multimodal experts, where only routing, filtering, and sampling decisions are allowed.

2. We propose C-QIN, a lightweight two-head planner trained from a limited counterfactual route bank to predict route utilities and select budget-aware retrieval plans.

3. We introduce calibrated hard filtering to control false elimination of ground-truth videos, and evaluate accuracy-cost-safety tradeoffs under noisy metadata and oracle route baselines.
```

### 14.3 Related work 必补方向

必须添加以下相关工作段落：

```text
- cascaded retrieval / multi-stage ranking / cost-sensitive learning to rank
- budgeted inference / conditional computation
- black-box or API-only adaptation / frozen expert routing
- query performance prediction
- query-aware frame selection
- metadata-aware multimedia retrieval
```

重点区分：

```text
已有 cascade ranking:
  多数关注文档排序管线或 learned ranker stage。

本文:
  冻结多模态专家不可改，route action 跨 semantic/time/geo/motion/device、hard/soft filter、candidate size、model call budget。
```

### 14.4 Theory section

建议不要单独设很大的 theory section。可以设：

```text
Calibration and Risk Control
```

写法：

```text
We use a held-out calibration split to select per-axis safety thresholds. Under exchangeability between calibration and test queries, conformal-style calibration provides finite-sample control of false hard-filter activation. In practice, we also report stress tests under metadata noise and distribution shift.
```

不要夸大为：

```text
We theoretically guarantee retrieval performance under arbitrary query shifts.
```

---

## 15. 禁止事项清单

Claude Code 修改时必须避免：

```text
[ ] 不要把 69.5% synthetic metadata 写成真实主结果。
[ ] 不要声称 Ego4D 有真实 GPS，除非代码确认字段存在。
[ ] 不要实现 4-head QIN 作为默认主方法。
[ ] 不要把平凡 regret bound 放在主贡献里。
[ ] 不要在 test set 上选择 calibration threshold。
[ ] 不要让 hard filter axis 与 soft axis 重叠。
[ ] 不要声称 energy 是 measured，除非有真机测量。
[ ] 不要删除已有 Phase 2 / Phase 3 baseline；必须保留用于对比。
[ ] 不要修改 frozen expert 模型权重。
[ ] 不要把 self-defined BBVR-Bench 当作大 benchmark 主贡献，除非公开数据和 protocol。
```

---

## 16. 推荐执行顺序

### Milestone 1：一周内应完成

```text
1. repo audit
2. baseline snapshot
3. route schema
4. route bank config
5. route executor basic implementation
6. unit tests for route bank
```

### Milestone 2：两到三周内应完成

```text
1. route bank builder
2. noisy metadata injection
3. oracle route generation
4. baseline B0-B5
5. main route evaluation CSV
```

### Milestone 3：三到五周内应完成

```text
1. C-QIN model
2. C-QIN training
3. calibration threshold selection
4. calibrated planner inference
5. baselines B6-B7
6. ablation table
```

### Milestone 4：后续增强

```text
1. cascade/cost-sensitive baseline
2. LLM parser baseline
3. YFCC100M or RealAlbum real metadata subset
4. QVHighlights / Charades-STA moment localization
5. true device power / NPU active measurement
```

---

## 17. Claude Code 首条执行 Prompt

可以直接把下面这段发给 Claude Code：

```text
You are modifying the LiteVTR++ repository to make it suitable for an AAAI-style submission.

Do not implement a 4-head QIN. Implement a minimal 2-head Calibrated Query-Intent Network (C-QIN):
1. route_value_head: predicts utility for each route in a fixed ~30-route bank;
2. filter_safety_head: predicts per-axis GT survival probability for hard filtering.

First audit the repository and identify paths for QueryParser, MetaFilter, OfflineIndex, QPP planner, reranking, col-softmax, and evaluation scripts. Then create a baseline snapshot before modifying behavior.

Add modules for:
- route schema and route bank config;
- counterfactual route bank builder;
- noisy metadata injection;
- C-QIN model/training;
- calibration threshold selection;
- calibrated planner inference;
- baselines: semantic-only, rule parser, QPP-only, random route, oracle route, always-hard-filter-all, C-QIN without calibration, C-QIN with calibration;
- metrics: R@K, MeanR, MRR, GT filtered rate, hard-filter activation rate, model calls/query, ms/query, oracle gap.

Critical constraints:
- all expert models remain frozen;
- never use perfect synthetic 69.5% R@1 as the main result;
- mark perfect synthetic metadata as oracle upper bound only;
- do not select calibration thresholds on test data;
- report GT filtered rate for every hard-filter method;
- if energy is not actually measured, label it as proxy.

After each milestone, run tests and export CSV reports under reports/aaai_main/.
```

---

## 18. 最终交付物清单

Claude Code 完成后，项目应至少新增或修改：

```text
Code:
  litevtr/routing/route_schema.py
  litevtr/routing/route_bank.py
  litevtr/routing/route_executor.py
  litevtr/routing/route_bank_builder.py
  litevtr/routing/qin_model.py
  litevtr/routing/train_qin.py
  litevtr/routing/calibrate_safety.py
  litevtr/routing/calibrated_planner.py
  litevtr/routing/baselines.py
  litevtr/metadata/noisy_metadata.py
  litevtr/eval/metrics.py
  litevtr/eval/eval_planner.py

Configs:
  configs/route_bank_30.yaml
  configs/qin_train.yaml
  configs/metadata_noise_msrvtt.yaml
  configs/eval_main.yaml

Scripts:
  scripts/build_route_bank.sh
  scripts/train_cqin.sh
  scripts/calibrate_cqin.sh
  scripts/eval_cqin.sh
  scripts/make_aaai_tables.sh

Reports:
  reports/baseline_snapshot/*
  reports/route_bank/*
  reports/aaai_main/main_results.csv
  reports/aaai_main/ablation_results.csv
  reports/aaai_main/calibration_results.csv
  reports/aaai_main/oracle_gap.csv
  reports/aaai_main/metadata_noise_sweep.csv

Tests:
  tests/routing/test_route_schema.py
  tests/routing/test_route_bank.py
  tests/routing/test_route_executor.py
  tests/routing/test_qin_model.py
  tests/routing/test_calibration.py
  tests/eval/test_metrics.py
```

---

## 19. 投稿判断标准

实现完成后，按下面标准判断是否继续冲 AAAI 主会。

### 可以冲 AAAI 的条件

```text
- C-QIN calibrated 在 realistic noisy metadata 上优于 rule parser / QPP-only / always-hard-filter-all；
- GT filtered rate 明显低于 always-hard-filter-all 和 uncalibrated C-QIN；
- oracle gap 足够小，说明 learned planner 接近 route bank 上限；
- synthetic 69.5% 已经降级为 upper bound；
- 至少有一个 convincing 的 noisy metadata 或真实 metadata 实验；
- 相关工作能清楚区分 cascade ranking / budgeted inference / frozen expert routing。
```

### 应考虑转投 MLSys / ACM MM / SIGIR 的情况

```text
- C-QIN 提升不明显，但系统加速、真机效率、cache/scheduler 贡献很强；
- calibrated hard filtering 只在 synthetic 设置有效；
- route learning 不能稳定超过 rule parser；
- 真实 metadata 或 noisy metadata 实验不足以说服 AAAI 审稿人；
- 主要贡献仍然是工程 pipeline，而不是 learned calibrated planning。
```

---

## 20. 最终原则

本次修改不要追求“模块越多越强”。AAAI 主会需要的是清晰、可验证、可解释的 novelty。

最强主线只有一句：

> **C-QIN learns when and how to route frozen multimodal experts under a budget, while calibrated hard filtering controls the risk of deleting the correct video.**

所有代码、实验、表格、论文文本都应该服务于这句话。
