# CSER 最终修改方案：MSRVTT10K 全量实验前修正版

> 本方案根据组内评审文件 `CSER_MODIFICATION_PLAN_REVIEW_AND_CORRECTED_IMPLEMENTATION.md` 修正而来，替代上一版 `CSER_NEXT_MODIFICATION_PLAN_FOR_MSRVTT10K.md` 作为后续讨论与实现依据。  
> 本文件只定义修改方案，不代表当前已经完成代码实现。

## 1. 最终结论

当前 CSER 的主要可保留贡献是 **conformal safety filtering/reporting**，而不是已经被证明的 retrieval accuracy gain。Phase1/Phase2 显示 learned selector 仍不稳定：budget 增大没有带来 value 提升，且 learned greedy value 低于 semantic-only。因此下一阶段的核心目标不是直接跑 MSRVTT10K final，而是先修正 selector、锁定 dev/final protocol、完成 candidate top-k sweep 与 10K gallery cache。

最终方案采用以下关键修正：

| 决策 | 最终方案 |
|---|---|
| 主 selector 方向 | 新增 `SetValueNetwork`，直接预测 `F(q,S)` 并枚举 feasible subset |
| fallback | 只保留 `min_delta` 一层；删除 validation-negative mask blocklist |
| min_delta sweep | `0.0 / 0.001 / 0.002 / 0.005` |
| safety 实验语义 | 拆成 `*_reduce` 与 `*_report` 两类配置分别报告 |
| monotone envelope | 从主计划删除；不再用 envelope 训练 raw retrieval selector |
| no_face_id | 从 P1 提升为 P0 |
| 实验规模 | 1K 最多约 50 configs；10K dev 跑 12 configs；10K final 只跑 1-2 configs |
| 38.8% 目标数字 | 不作为论文目标写入；最终只报告 delta 与完整 baseline 表 |

## 2. P0 修改项

### 2.1 修正 selector：SetValueNetwork + safe fallback

**修改内容**

新增 `SetValueNetwork` 作为主修正方向。它不再学习 marginal value `v(e|S,q)`，而是直接学习每个 subset 的最终检索价值：

```text
query_feat -> [F(q,S0), F(q,S1), ..., F(q,S15)] -> feasible subset argmax
```

由于当前 optional experts 为 4 个，subset 总数只有 16 个，推理时可以直接枚举所有 budget-feasible masks。训练目标直接使用 `OracleLabels.value_matrix`，无需修改 `value_oracle.py`。

保留四种 selector mode 做对比：

| Selector | 用途 |
|---|---|
| `marginal_value_greedy` | 当前 SVN greedy，作为 legacy baseline |
| `marginal_density_greedy` | 按 `predicted_marginal / cost` 选择，检验 cost-aware greedy 是否足够 |
| `set_value` | 直接选择预测 `F(q,S)` 最高的 feasible subset |
| `set_value_safe` | 推荐候选策略；若预测收益不超过 semantic-only，则退回 semantic-only |

`set_value_safe` 的 fallback 只保留一层：

```text
if predicted_best <= predicted_empty + min_delta:
    use semantic-only
```

明确删除以下规则：

```text
if selected_mask has negative mean validation delta:
    use semantic-only
```

**修改原因**

validation-negative mask blocklist 来自 dev split，直接用于 test/final 会引入 train/test mismatch。如果某个 mask 在 dev 上负收益、但在 test 上正收益，该规则会系统性伤害泛化能力。相比之下，`min_delta` 只依赖模型对当前 query 的预测，不依赖 dev 上的 mask 统计，风险更低。

SetValueNetwork 的原因是：当前 marginal greedy 依赖子模性和逐步边际预测，但实测存在 monotonicity violation，且 greedy 误差会累积。直接预测 set value 可以绕过子模假设，以数据驱动方式学习最优 expert subset。

**需要特别澄清**

`set_value_safe` 只能保证“预测值”不低于 semantic-only，不保证真实 R@1 一定不低于 semantic-only。真实效果必须通过 dev/test 实证验收。

**验收标准**

- 在 1K/dev 上，`set_value_safe` 的 R@1/MRR 不低于 semantic-only 超过 `0.5pp` 容忍线。
- budget 增大时 mean value 不应下降超过 `0.002`。
- `fallback_triggered_rate` 必须报告；若超过 50%，优先检查 SetValueNetwork 预测质量，而不是继续调高 `min_delta`。
- 以同一配置头对头比较：
  - `marginal_value_greedy` vs `set_value_safe`
  - `all` vs `no_face_id`

### 2.2 selector 诊断输出

**修改内容**

必须新增三类诊断文件：

| 文件 | 内容 |
|---|---|
| `e1_cser_query_audit.jsonl` | 每条 query 的 selected mask、semantic-only rank/RR、CSER rank/RR、oracle mask/rank/RR、delta、candidate count、gt filtered |
| `selector_mask_distribution.json` | mask 频率、expert 选择频率、每个 mask 的平均 RR/R@1、相对 semantic-only delta |
| `expert_delta_summary.json` | 每个 expert 单独加入、被 selector 选中、被 oracle 选中时的真实 delta |

**修改原因**

当前仅有 aggregate metric，无法判断 selector 失败来自 policy collapse、某个 expert 噪声、某类 query 误判，还是训练目标不匹配。query-level audit 是修正 selector 前的必要诊断。

**验收标准**

- 能抽样解释三类 query：CSER 优于 semantic-only、CSER 差于 semantic-only、oracle 明显优于 CSER。
- mask distribution 不应显示绝大多数 query 都选择同一个负收益 mask。
- `expert_delta_summary.json` 能支持是否禁用 `face_id` 或修复 `scene` 的决策。

### 2.3 no_face_id 提升为 P0

**修改内容**

将 `no_face_id` 从 P1 提升为 P0，并在 selector 修正实验开始前纳入最小矩阵。

必测 expert roster：

| Roster | 说明 |
|---|---|
| `all` | 当前完整 optional experts |
| `no_face_id` | 禁用 face_id，保留 highlight / face / scene |
| `semantic_highlight_scene` | 更轻量候选，用于补充 ablation |

**修改原因**

MSRVTT text-to-video 通常没有 reference face embedding，`face_id` 很可能高噪声、高成本、低收益。若在 selector 修正前不确认默认 roster，后续 selector 消融会被无效 expert 干扰。

**验收标准**

- 如果 `no_face_id` 不降低 R@K 且降低 avg cost，则作为 MSRVTT10K 主表默认 roster。
- 如果 `all` 更好，必须通过 expert contribution table 说明 `face_id` 的真实贡献。

### 2.4 safety filtering/reporting 拆分 reduce 与 report

**修改内容**

将 conformal safety 配置拆成两类，避免“过滤”语义混淆：

| 配置类型 | 行为 | 报告重点 |
|---|---|---|
| `Mondrian_reduce` / `Split_reduce` | 实际 hard filter：保留 `semantic_top_k(q) union C(q)`，删除其补集 | R@K、GT_filtered_rate、candidate_reduction_rate |
| `Mondrian_report` / `Split_report` | 不删除候选，只在 full gallery 上报告 coverage | conformal_coverage、GT_in_set_rate |

`reduce` 模式的候选集合定义为：

```text
final_candidates(q) = semantic_top_k(q) union C(q)
```

`no_gate` baseline 只保留 `semantic_top_k(q)`；`no_filter` baseline 使用 full gallery。

**修改原因**

`C(q)` 是应该保留的 conformal set，而不是直接要删除的集合。只有明确 `reduce` 与 `report` 的差异，才能正确解释 candidate reduction 与 coverage 结果。

**验收标准**

- 所有 safety 表必须标注配置是 `reduce` 还是 `report`。
- `reduce` 表必须报告 `GT_filtered_rate`、candidate count p50/p90/p95/p99。
- `report` 表不得声称产生 candidate reduction。

### 2.5 candidate_top_k sweep

**修改内容**

dev protocol 中测试：

```text
candidate_top_k = 100 / 300 / 500 / 1000 / 2000(optional)
```

每个 top-k 至少比较：

- `no_gate`
- `Mondrian_reduce`
- `Split_reduce`
- `heuristic_reduce`

**修改原因**

1K 中 top-100 保留 10% gallery；10K 中 top-100 只保留 1% gallery。不能把 1K top-k 直接迁移到 10K。

**验收标准**

配置选择规则固定为：

```text
GT_filtered_rate <= alpha
R@1 不低于 no_filter / semantic-only 超过 0.5pp
在满足前两项的配置中 avg_candidates_after_filter 最小
```

### 2.6 10K gallery expert cache 与失败视频处理

**修改内容**

建立可复用 gallery expert cache，Phase1/2/3 共用同一份 cache。`manifest.json` 必须记录：

- `n_videos_total`
- `n_videos_loaded`
- `failed_video_ids`
- expert class names
- checkpoint paths
- feature shapes / dtype
- created_at
- whether mock fallback occurred

失败视频处理规则必须统一：

- `failed_video_ids` 在所有 phase 中一致读取。
- 保留 total gallery size 记录为 10000。
- scoring/ranking 只在有效 loaded videos 上执行；报告中同时写清 total videos、loaded videos、failed videos。
- Phase1 完成后打印：

```text
N videos failed: X; N videos loaded: Y; N videos total: 10000
```

**修改原因**

10K 上重复 expert extraction 成本高且易失败。失败视频如果在不同 phase 中处理不一致，会导致不可复现的指标偏差。

**验收标准**

- 第二次运行 Phase1/2/3 不再重复大规模 expert extraction。
- `--real-models` 下没有 mock fallback；若开启 fail-closed，真实模型初始化失败直接停止。
- manifest 中的 failed video 处理与实际 loaded gallery 完全一致。

### 2.7 dev/final protocol 固定

**修改内容**

优先使用 MSRVTT10K 官方 split：

```text
dev split   = official val set
final split = official test set
```

如果没有官方 val split，则按 query 顺序固定切分：

```text
dev split   = first 60% queries
final split = last 40% queries
```

所有 selector、top-k、gate、roster、min_delta 调参只能在 dev split 完成。final split 只跑锁定配置。

**修改原因**

避免在 final test 上反复调参后报告最优结果。

**验收标准**

- final run 前配置写入 manifest。
- final test 上不做模型选择。
- final report 明确 train/cal/dev/final 的 query 数量和来源。

## 3. P1 修改项

### 3.1 latency 与 cost scope

保留上一版方案，必须区分三类 timing：

1. Offline build time：decode、MobileCLIP、MomentDETR、SCRFD、ArcFace、MobileNetV3。
2. Query-time rerank time：semantic score、candidate top-k、conformal mask、cached score fusion。
3. End-to-end amortized time：offline build time amortized over query count + query-time rerank。

当前 `avg_cost` 必须标为：

```text
cost_kind = offline_index_expert_unit_proxy
```

不能把 cached rerank speedup 写成端到端专家模型加速。

### 3.2 scene expert mapping audit

将 scene audit 改成两阶段流程：

1. 自动化阶段：统计每个 scene 类别的 query coverage、GT match rate、单独加入 scene expert 后的 delta RR。
2. 选择性人工阶段：只对 match rate 极低（建议 `<20%`）的类别人工检查；人工检查只用于确认 vocabulary mapping 问题，不作为性能指标。

## 4. 删除或降级项

### 4.1 删除 monotone envelope 主计划

删除上一版 P1.1 monotone envelope objective。原因是 envelope 训练 + raw retrieval 评测会引入新的 train/test objective mismatch，而且 SetValueNetwork 已经绕过对子模性假设的依赖。

最终理论叙事调整为：

```text
CSER 不再依赖严格 monotone submodular 假设来获得主 selector；
我们使用数据驱动的 set-value predictor 直接学习 budget-feasible expert subset。
子模性分析保留为诊断与解释，而不是主方法成立的必要条件。
```

如果后续仍想保留 envelope，只能作为独立 ablation，并且必须在 envelope metric 上评测，不能用 raw metric 声称 selector 改进。

### 4.2 删除 38.8% 目标数字

如果 MSRVTT10K final R@1 达不到 38.8%，该数字必须从论文计划中删除。论文不写固定 R@1 目标，只报告：

- CSER vs semantic-only 的 delta。
- CSER 与所有 baselines 的完整绝对数值。
- candidate reduction 与 GT_filtered tradeoff。

不得单独报告 CSER 的目标绝对数字而不报告全部 baselines。

## 5. 实验矩阵

完整矩阵约为：

```text
selector x top-k x gate x roster x value transform
```

规模过大，不现实。最终执行规模如下：

| 阶段 | 规模 | 目的 |
|---|---:|---|
| 1K/dev | 最多约 50 configs | 看趋势、筛掉明显差的策略 |
| 10K/dev | 12 configs | 锁定最终 selector/top-k/gate/roster |
| 10K/final | 1-2 configs | 最终报告，不再调参 |

10K/dev 最小必跑矩阵：

| candidate_top_k | gate | roster | selector |
|---:|---|---|---|
| 100 | no_gate | all | current learned |
| 100 | Mondrian_reduce | all | current learned |
| 300 | Mondrian_reduce | all | current learned |
| 500 | Mondrian_reduce | all | current learned |
| 1000 | Mondrian_reduce | all | current learned |
| 500 | Mondrian_reduce | all | `marginal_value_greedy` |
| 500 | Mondrian_reduce | all | `set_value_safe` |
| 500 | Split_reduce | all | current learned |
| 500 | heuristic_reduce | all | current learned |
| 500 | Mondrian_reduce | no_face_id | current learned |
| 500 | Mondrian_reduce | no_face_id | `marginal_value_greedy` |
| 500 | Mondrian_reduce | no_face_id | `set_value_safe` |

每个配置必须报告：

```text
R@1, R@5, R@10, MRR,
GT_filtered_rate, conformal_coverage,
avg_candidates_after_filter, candidate_reduction_rate,
avg_cost, avg_experts_called,
candidate count p50 / p90 / p95 / p99,
selector mask distribution summary,
fallback_triggered_rate (set_value_safe only)
```

## 6. 实现清单

后续实现阶段按以下清单推进：

- [ ] 新增 `cser/set_value_network.py`：`SetValueNetwork`，直接预测 16 个 subset values。
- [ ] 新增 `cser/train_set_value.py`：使用 `OracleLabels.value_matrix` 训练 set-value predictor，支持多 seed。
- [ ] 新增或扩展 `cser/selectors.py`：`SetValueSelector`、`MarginalDensitySelector`、selector factory。
- [ ] 修改 Phase2 driver：支持 `--selector`、`--selector-model`、`--min-delta`、`--candidate-top-k-list`、`--expert-roster`。
- [ ] 新增 query audit、mask distribution、expert delta summary 输出。
- [ ] 实现 `*_reduce` 与 `*_report` 两类 safety 配置。
- [ ] 增加 gallery cache 读写与 manifest。
- [ ] 增加 real-model fail-closed preflight。
- [ ] 在 1K/dev 跑 12-config 最小矩阵，随后扩展到最多约 50 configs。
- [ ] 在 10K/dev 跑 12-config 最小矩阵。
- [ ] 锁定 final protocol 后，只在 10K/final 跑 1-2 个配置。

## 7. 最终进入 MSRVTT10K final 的 gate

只有满足以下条件，才进入 final run：

- query audit 已跑通。
- `set_value_safe` 与 `marginal_value_greedy` 已在相同配置下头对头比较。
- `no_face_id` 已作为 P0 roster 比较完成。
- top-k sweep 已在 dev 完成。
- real-model preflight 通过，无 mock fallback。
- 10K gallery cache 可复用，failed videos 处理一致。
- final selector/top-k/gate/roster/min_delta 已写入 manifest。
- final 不再调参。
- 如果 selector 修正后 R@1 仍不超过 semantic-only + 2pp，论文标题和 contribution statement 应从 “improved retrieval accuracy” 调整为 “efficient expert routing with conformal safety”。

## 8. 最终论文叙事建议

建议把 AAAI contribution statement 调整为：

1. `set_value` selector 在 budgeted expert selection 中提供比 marginal greedy 更稳定的经验改进。
2. conformal safety gate 在 10K scale 上实现可测的候选缩减，并控制 GT false elimination。
3. 方法在真实 expert cache 与固定 dev/final protocol 下可复现。

暂不建议使用：

```text
CSER improves retrieval accuracy over all baselines.
```

除非 final 表中 CSER 对 semantic-only、UCB、cascade 等 baselines 均有稳定优势。

