# CSER MSRVTT10K Phase2 实验结果分析与失效原因讨论

> 作业号：`9814419`  
> 数据集：MSR-VTT 10K，10,000 videos / 10,000 queries  
> 分析日期：2026-06-11  
> 对照计划：`docs/delivery/CSER_FINAL_MODIFICATION_PLAN_FOR_MSRVTT10K.md`

## 1. 执行摘要

本次实验只能得出以下两个可靠结论：

1. **Conformal safety filtering 达到预期。** 在 `alpha=0.05` 下，实测 coverage 为 `96.32%`，GT filtered rate 为 `3.68%`，同时将平均候选数从 10,000 降至约 2,168--2,319，candidate reduction 为 `76.8%--78.3%`。
2. **已完成的两个 legacy selector 均未达到 retrieval 预期。** `marginal_value_greedy` 和 `marginal_density_greedy` 的 R@1、R@5、R@10、MRR 均显著低于 semantic-only，也低于 UCB baseline。

但是，本次作业**不能用于判断最终修改计划中的主方案 `set_value_safe` 是否有效**。作业只完整完成：

- `marginal_value_greedy + all experts`
- `marginal_density_greedy + all experts`

在启动第一个 `set_value_safe + all experts` 配置时，进程因 ONNX Runtime 缺少 `CUDAExecutionProvider` 而失败。因此以下计划中的关键实验均没有结果：

- `set_value_safe` 与 legacy greedy 的同配置比较；
- `all` 与 `no_face_id` 的比较；
- `semantic_highlight_scene` roster；
- `min_delta` sweep；
- `set_value_safe` 的 fallback rate；
- 10K/dev 最小 12-config 矩阵；
- Phase3。

综合现有证据，失效原因按优先级判断为：

1. **专家信号和固定加法融合存在严重失真，尤其是 scene expert；**
2. **legacy SVN selector 的预测目标、正则与决策规则不能可靠识别少量真正受益的 query；**
3. **作业流程未执行到计划主方案，且实验期间代码版本发生变化；**
4. **dev/final protocol、模型复用和 cache-only 加载路径未按计划落实；**
5. **GPU/ONNX Runtime 配置造成运行时间过长和主实验中断，但不是 retrieval 指标下降的直接原因。**

因此，本次实验总体评价为：

> **Safety 子目标达标；retrieval 子目标未达标；最终 selector 子目标未完成，不能验收；当前结果支持把论文贡献暂时收缩为“efficient expert routing with conformal safety”，而不能声称提升了 retrieval accuracy。**

## 2. 作业实际完成状态

### 2.1 时间线

| 阶段 | 开始时间 | 结束时间 | 用时 |
|---|---:|---:|---:|
| 作业启动与 preflight | 2026-06-06 11:38 | 2026-06-06 11:40 | 约 2 分钟 |
| Phase1 | 2026-06-06 11:40 | 2026-06-10 07:49 | 约 92.1 小时 |
| Phase2 marginal value | 2026-06-10 07:49 | 2026-06-10 16:09 | 约 8.3 小时 |
| Phase2 marginal density | 2026-06-10 16:09 | 2026-06-11 00:18 | 约 8.2 小时 |
| Phase2 set value safe | 2026-06-11 00:18 | 2026-06-11 00:19 | 初始化失败 |

最终状态文件为：

```text
state=failed
detail=cser_phase2_set_value_safe_all
exit_code=1
```

所以“Phase2 已完成”只能理解为两个 legacy 配置完成，不能理解为计划定义的完整 Phase2 矩阵完成。

### 2.2 失败点

`set_value_safe_all` 在读取任何数据和训练 SetValueNetwork 之前失败：

```text
RuntimeError: InsightFace requested GPU execution, but ONNX Runtime has no
CUDAExecutionProvider; available providers:
['AzureExecutionProvider', 'CPUExecutionProvider']
```

完整 gallery cache 已经存在，但 `cser/data.py::load_video_dataset` 在检查 cache 之前先执行：

```python
bundle = build_model_bundle(use_real=use_real_models)
```

因此即使本阶段只需读取缓存特征，也必须重新初始化 InsightFace、MomentDETR、MobileNet 等模型。这个加载顺序使一个与 selector 评测无关的 GPU provider 问题阻断了整个实验。

## 3. 核心结果

### 3.1 主结果对比

测试集为随机切分得到的 2,500 queries，budget=5，主 safety 配置为 `Mondrian_reduce`。

| 方法 | R@1 | R@5 | R@10 | MRR | Avg cost | Avg experts |
|---|---:|---:|---:|---:|---:|---:|
| Semantic-only | **14.36%** | **28.96%** | **36.60%** | **0.2188** | 1.00 | 1.00 |
| All experts | 12.88% | 24.84% | 31.20% | 0.1921 | 5.00 | 3.00 |
| Random | 12.72% | 25.88% | 32.76% | 0.1954 | 3.44 | 2.25 |
| UCB | 13.52% | 27.40% | 34.52% | 0.2065 | 4.28 | 2.35 |
| Marginal value CSER | 13.04% | 26.36% | 33.12% | 0.1988 | 3.06 | 2.10 |
| Marginal density CSER | 13.00% | 26.16% | 32.96% | 0.1982 | 2.98 | 2.10 |
| Oracle | **17.60%** | **31.60%** | **39.60%** | **0.2486** | 2.03 | 1.52 |

相对 semantic-only：

| Selector | Delta R@1 | Delta MRR |
|---|---:|---:|
| Marginal value | **-1.32pp** | **-0.0200** |
| Marginal density | **-1.36pp** | **-0.0206** |
| Oracle | **+3.24pp** | **+0.0299** |

Oracle 明显高于 semantic-only，说明 optional experts 并非完全无用，理论上存在 query-dependent routing 空间；但所有固定策略和已完成的 learned selector 都低于 semantic-only，说明系统不能可靠识别这些稀疏的正收益 query。

### 3.2 配对统计

基于 2,500 条相同测试 query，使用 5,000 次 paired bootstrap：

| Selector | R@1 delta 95% CI | MRR delta 95% CI |
|---|---:|---:|
| Marginal value | `[-2.04pp, -0.60pp]` | `[-0.0259, -0.0143]` |
| Marginal density | `[-2.08pp, -0.64pp]` | `[-0.0265, -0.0147]` |

两个 selector 的下降都不是简单随机波动。

Query-level 分布进一步显示：

| Selector | 改善 query | 变差 query | 不变 query |
|---|---:|---:|---:|
| Marginal value | 18.84% | 36.60% | 44.56% |
| Marginal density | 16.68% | 35.52% | 47.80% |

Marginal density 相比 marginal value 只减少了少量 cost，没有改善 retrieval，说明问题不是简单的 cost-aware 排序缺失，而是 predicted marginal 的符号、尺度和 query 条件判断本身不可靠。

## 4. 与计划验收标准的逐项对照

| 计划验收项 | 本次结果 | 判断 |
|---|---|---|
| `set_value_safe` R@1/MRR 不低于 semantic-only 超过 0.5pp | `set_value_safe` 未运行 | **无法验收** |
| Budget 增大时 mean value 下降不超过 0.002 | MRR 从 B=1 的 0.2188 降至 B=3 的 0.1998 | **失败** |
| 报告 fallback rate | 仅 legacy selector，无 safe fallback | **未完成** |
| marginal value vs set value safe | 后者无结果 | **未完成** |
| all vs no_face_id | `no_face_id` 未运行 | **未完成** |
| Query audit / mask distribution / expert delta | 三类文件均生成 | **通过** |
| Safety reduce/report 语义分离 | E6 已区分并正确报告 | **通过** |
| GT filtered rate <= alpha | 3.68% <= 5% | **通过** |
| Candidate percentile 与 reduction 报告 | 已报告 p50/p90/p95/p99 | **通过** |
| Top-k 配置 R@1 不低于 semantic-only 超过 0.5pp | 所有 legacy reduce 配置低 1.32pp | **失败** |
| 10K cache 可复用、无 mock fallback | 10,000/10,000，failed=0 | **通过** |
| Official dev/final 或固定顺序 60/40 | 实际为随机 60/15/25 | **失败** |
| 10K/dev 12 configs | 完成 2 个 legacy configs | **失败** |
| Final 前锁定 manifest | 无 final manifest | **未完成** |
| 多 seed | 仅 seed=42 | **未完成** |

结论是：**计划中的安全和诊断模块已经得到支持，但 selector 修正与实验协议没有完成，因此整体实验未达到预期。**

## 5. Safety 模块分析

### 5.1 Coverage

| Alpha | Target coverage | Split | Mondrian |
|---:|---:|---:|---:|
| 0.01 | 99% | 99.24% | 99.16% |
| 0.05 | 95% | 96.32% | 96.32% |
| 0.10 | 90% | 91.00% | 91.00% |
| 0.20 | 80% | 80.52% | 80.84% |

四个 alpha 下 empirical coverage 都不低于目标，说明 conformal calibration 在当前随机切分上工作正常。

### 5.2 Candidate reduction

Alpha=0.05、candidate top-k=500：

| 配置 | GT filtered | Coverage | Avg candidates | Reduction |
|---|---:|---:|---:|---:|
| No gate | 16.56% | N/A | 500 | 95.00% |
| Mondrian reduce | 3.68% | 96.32% | 2,319 | 76.81% |
| Split reduce | 3.68% | 96.32% | 2,168 | **78.32%** |
| Heuristic reduce | 2.60% | 97.40% | 2,881 | 71.19% |
| Report / no filter | 0% | 96.32% / N/A | 10,000 | 0% |

在 safety 子任务上，Split conformal 比 Mondrian 使用更少候选，并保持相同 coverage 和 GT filtered rate。现有结果中，`Split_reduce` 是更高效的 safety 配置。

### 5.3 Top-k sweep

No-gate 的 GT filtered rate：

| Top-k | GT filtered |
|---:|---:|
| 100 | 36.04% |
| 300 | 21.64% |
| 500 | 16.56% |
| 1000 | 10.00% |

即使 top-k=1000，no-gate 仍远高于 alpha=5%，说明 10K gallery 中不能直接沿用小规模 top-k。

加入 conformal union 后，top-k=100--1000 的 R@1 基本不变，平均候选数只从约 2,307 增至 2,381。这说明最终候选集合主要由 conformal set 决定，semantic top-k 在当前 alpha 下不是主要瓶颈。

但是，所有 top-k 下 legacy selector 的 R@1 都约为 13.04%，低于 semantic-only 14.36%。因此**没有一个 top-k 配置同时满足 safety 和 retrieval 两项选择规则**。

## 6. Selector 失效分析

### 6.1 Budget 曲线直接违反预期

| Budget | CSER R@1 | CSER MRR | Avg cost |
|---:|---:|---:|---:|
| 1.0 | 14.36% | 0.2188 | 1.00 |
| 3.0 | 13.16% | 0.1998 | 2.88 |
| 5.0 | 13.04% | 0.1988 | 3.06 |
| 7.0 | 13.00% | 0.1984 | 3.17 |
| 9.5 | 13.00% | 0.1985 | 3.46 |

允许 selector 调用专家后，性能立即下降；继续增加预算只增加 cost，不恢复 retrieval。这与“更多预算允许更好 expert subset”的预期相反。

需要注意：当前 `exp_e4` 没有接收实际配置的 selector，而是内部使用默认 legacy greedy。因此 marginal-density 目录中的 E4 与 marginal-value 完全相同，未来即使运行 `set_value_safe`，现有 E4 代码也不能正确测量新 selector 的 budget curve。这是实现缺陷，不应把该曲线用于比较不同 selector。

### 6.2 低训练 MSE 不等价于正确 routing

SVN 的 validation MSE 约为 `0.00158`，但 routing 明显失败。原因包括：

- 最终决策是多个 predicted values 的 argmax，微小回归误差可能改变 mask；
- 正收益 query 稀少，MSE 容易被大量零收益或负收益样本主导；
- 决策关注的是 subset regret 和排序，而不是全体 marginal 的平均平方误差；
- Greedy 会逐步累积前一步的预测错误；
- Legacy selector 没有 semantic-only safe fallback。

在 budget=5 的 oracle 中，`55.48%` query 的最佳决策就是 semantic-only。模型如果不能准确判断剩余 44.52% 中哪些 query 真正受益，主动调用专家会产生负期望收益。

### 6.3 Ablation 暗示结构和正则不匹配

| SVN variant | R@1 | MRR | Avg cost |
|---|---:|---:|---:|
| Full | 13.04% | 0.1988 | 3.06 |
| No cross attention | 13.00% | 0.1987 | 2.94 |
| No set conditioning | **14.12%** | **0.2128** | **1.19** |
| No submod loss | 13.20% | 0.2038 | 2.49 |
| Semantic-only | **14.36%** | **0.2188** | 1.00 |

两个重要信号：

1. 删除 set conditioning 后性能显著改善，主要因为模型接近“少调用专家”，而不是更好地选择专家。
2. 删除 submodular loss 后优于 full model，说明强制预测满足子模结构会伤害 raw retrieval objective。

这些结果与 Phase1 的数据性质一致：

- monotonicity violation：20.475%；
- submodularity violation：5.910%；
- raw retrieval value 并不满足可靠的单调子模假设。

因此，legacy marginal greedy 理论路线在当前 expert score 定义下缺乏充分支持。该结论不否定 conformal theory，也不直接否定尚未运行的 SetValueNetwork。

## 7. Expert 信号失效分析

### 7.1 Expert 的真实平均贡献

| Expert | Solo delta MRR | 正收益 query | 负收益 query | Oracle 选择率 |
|---|---:|---:|---:|---:|
| Highlight | -0.00147 | 34.20% | 42.92% | 34.04% |
| Face | -0.02468 | 16.60% | 25.48% | 16.24% |
| Face-ID | 0.00000 | 0% | 0% | 0% |
| Scene | -0.03820 | 1.72% | 26.72% | 1.52% |

Experts 并非对所有 query 单调有益。它们只在少数 query 上产生较大正收益，同时在更多 query 上改变正确的 semantic 排序。

### 7.2 Scene expert 是当前最严重的数据/模型失配

进一步 audit 得到：

- Gallery dominant scene 中 `other` 占 `86.39%`；
- 具有 scene cue 的测试 query 共 774 条；
- Query scene cue 与 GT video dominant scene 的匹配率只有 `6.33%`；
- Scene solo delta MRR 为 `-0.0382`；
- Scene 只有 `1.72%` query 为正收益；
- Oracle 只在 `1.52%` query 选择 scene；
- Marginal-value selector 选择 scene `26.88%`；
- Marginal-density selector选择 scene `44.88%`。

当前 scene backbone 在没有 Places365 checkpoint 时使用 ImageNet MobileNetV3，再通过 object-label keyword mapping 映射到 12 个 scene bucket。ImageNet 的目标是物体分类，不是场景识别，大量视频最终映射为 `other`，query vocabulary 与视频标签体系严重不一致。

这不是 selector 能完全修复的问题。输入 expert signal 本身不可辨识时，更复杂的 selector 只会学习训练集中的偶然相关。

### 7.3 Face expert 被过度选择

Face solo delta MRR 为 `-0.0247`。Marginal-value selector 选择 face 的比例为 `49.16%`，但 oracle 只在 `16.24%` query 中选择 face。

当前 face expert 只是“视频是否存在高置信人脸”的 gallery prior。很多包含 `man/woman/person` 的 query 并不以脸部特写为判别条件，给所有有人脸的视频固定加分会稀释 semantic similarity。

### 7.4 Face-ID 在 MSRVTT text-to-video 中不可用

所有 query 的 `face_emb` prior 都为空，因此 Face-ID score 恒为零：

- Solo delta 为 0；
- Oracle 选择率为 0；
- 去除 Face-ID 后 oracle ceiling 完全不变。

`no_face_id` 应直接成为默认 roster，而不是继续作为需要验证其是否有效的开放问题。当前 legacy selector 选择 Face-ID 的比例不到 0.6%，因此仅去掉 Face-ID不会修复主要性能问题，但可以消除无意义的维度和成本定义。

### 7.5 Highlight 信号弱且 query conditioning 过粗

Highlight 是四个 optional experts 中最有潜力的一个，但平均贡献仍略为负。只有约 4.4% 测试 query 命中当前 highlight keyword；未命中时系统仍使用 `gain=0.3` 给高 highlight 视频加分。

这使 highlight 在多数 query 上退化为 query-agnostic popularity prior。需要更细粒度的动作匹配或直接学习 query-video highlight compatibility，而不是只使用十余个关键词。

### 7.6 固定加法融合放大 expert 噪声

当前 final score 为：

```text
semantic_score + 0.35 * sum(optional_expert_score)
```

所有 expert 使用统一固定权重，缺少：

- expert-specific calibration；
- query-specific confidence；
- score distribution calibration；
- 对负收益 expert 的抑制；
- 与 semantic margin/uncertainty 的交互。

`all_experts` 明显低于 semantic-only，是固定加法融合失效的直接证据。

## 8. 理论判断

本次实验不支持“当前 raw retrieval value 可以可靠按单调子模函数处理”：

- 20.475% monotonicity violations；
- 增加 budget 导致实际 value 下降；
- 移除 submod loss 反而改善 retrieval；
- Density greedy 未改善结果。

因此，原始 marginal-greedy 理论路线应继续降级为 legacy baseline。

但是，不能据此判断 SetValueNetwork 理论失败，因为它没有产生任何实验结果。计划中使用 set-value predictor 绕过严格子模假设的动机，反而被本次 legacy 结果进一步加强。

Conformal safety 的理论与实证则保持一致，在多个 alpha 下均达到目标 coverage。

## 9. 实验流程与复现性问题

### 9.1 数据划分未按计划落实

实际代码使用 seed=42 的随机 `60/15/25`：

```text
train=6000, calibration=1500, test=2500
```

计划要求优先 official val/test，缺失时使用 query-order 固定 `60/40` dev/final。当前 top-k 和 safety 对照都在同一个 test split 上反复运行，因此只能作为开发诊断，不能作为 final result。

### 9.2 运行期间代码版本发生变化

作业于 2026-06-06 启动，但关键文件在作业运行期间更新：

```text
2026-06-10 21:30  cser/train_svn.py
2026-06-10 21:35  cser/run_phase2.py
2026-06-10 21:35  cser/train_set_value.py
2026-06-10 21:35  tasks/real_models.py
2026-06-10 21:53  slurm/run_cser_msrvtt_updated.slurm
```

Marginal-density 配置当时仍在运行，随后 `set_value_safe` 新进程加载了更新后的 fail-fast 逻辑并失败。这意味着同一作业内不同配置不一定对应完全相同的代码快照。

此外，仓库存在大量未提交修改和未跟踪文件，报告中只能记录基础 commit `e3c9244a...`，无法从该 commit 单独重建本次实验。

研究组后续实验必须在提交前：

- 创建 immutable commit/tag；
- 将 commit SHA、diff status、环境 lock 写入 run manifest；
- 禁止在作业运行期间修改远端工作树。

### 9.3 Preflight 与实际运行要求不一致

作业启动时 preflight 明确显示 InsightFace 使用 `CPUExecutionProvider`，仍被判定通过。后续代码又要求 GPU provider，不存在时 fail-fast。

Preflight 必须验证最终运行所要求的 provider，而不能只验证“模型能初始化”。

### 9.4 完整 cache 不应触发模型初始化

当 cache manifest 显示：

```text
complete=true
n_videos_loaded=10000
failed_video_ids=[]
```

Phase2 selector evaluation 只需要 cached gallery signals 和 text encoder。当前代码仍初始化全部五个模型，使不相关依赖成为故障点，也浪费启动时间和 GPU memory。

### 9.5 每个配置重复进行无关训练

每个 Phase2 config 都会重新：

- 构造 train/cal/test 的 16-subset oracle lattice；
- 训练 production SVN；
- 训练 E5 的四个 SVN ablation；
- 执行 E3/E4/E6。

两个 legacy 配置分别耗时约 8 小时，但其中 E3、E4、E5 的大量结果完全重复。Production SVN 单次训练约 2.7 小时。

应把流程拆成：

1. 一次性生成 oracle labels；
2. 每个 seed 只训练一次模型；
3. selector/gate/top-k/roster 只做轻量 evaluation；
4. E5 ablation 单独运行，不嵌入每个主配置。

### 9.6 MeanR/MedR 不宜跨过滤配置直接比较

当前 MeanR 和 MedR 会排除 rank=-1 的 filtered GT，而 R@K/MRR 将其按失败计入。因此过滤越激进，MeanR 可能看起来越小，但不代表整体 retrieval 更好。

跨 safety 配置应优先比较 R@K、MRR、GT filtered rate 和 candidate count，不应以现有 MeanR/MedR 作为主要结论。

## 10. 失效原因优先级

### P0：Expert 表示和融合失真

证据强度：**高**

- Scene mapping 基本失效；
- Face 和 scene 平均为明显负收益；
- Face-ID 恒为零；
- All-experts 比 semantic-only 更差；
- Oracle 说明只有稀疏 query 适合调用 experts。

### P0：Selector 无法识别稀疏正收益 query

证据强度：**高**

- 两个 greedy selector 都显著低于 semantic-only；
- 低 MSE 未转化为低 policy regret；
- Face/scene selection rate 远高于 oracle；
- Budget 增加时 value 下降；
- Density correction 无效。

### P0：主方案未运行

证据强度：**确定**

- `set_value_safe` 无任何 artifact；
- `no_face_id`、min-delta、fallback 均无结果；
- 不能对最终修改方案做正面或负面验收。

### P1：理论假设与 raw retrieval objective 不匹配

证据强度：**高，限定于 legacy marginal route**

- Monotonicity violation 高；
- Submod loss 有负面影响；
- Greedy error 累积。

### P1：实验协议和实现路径不完整

证据强度：**确定**

- 随机 test split 上进行多配置比较；
- E4 selector 绑定错误；
- Cache-only 仍初始化模型；
- 代码版本未冻结；
- 重复训练造成巨大运行开销。

### P2：GPU/运行环境

证据强度：**确定，但属于间接原因**

- A40 已分配；
- InsightFace 在前半程实际使用 CPU provider；
- 后半程要求 GPU provider 后直接失败；
- GPU 问题导致运行慢和实验中断，但不能解释已完成 legacy 配置的 retrieval 指标下降。

## 11. 建议的下一轮最小实验

在继续大规模运行前，建议按以下顺序执行。

### 11.1 先修复工程与协议

1. 冻结 commit、环境和 run manifest。
2. 实现 cache-only dataset loader，完整 cache 下不初始化视觉模型。
3. 统一 ONNX Runtime 策略：要么确认 CUDA provider 可用，要么明确使用 CPU；preflight 与运行逻辑一致。
4. 使用 official split；若无法获得，严格执行 query-order dev/final。
5. 修复 E4，使其显式接收实际 selector。
6. Oracle labels 和训练模型持久化复用。
7. 主实验不再自动重复 E5 ablation。
8. 至少运行 3 个 seeds，并报告 mean/std。

### 11.2 先缩小 roster

建议优先测试：

| Roster | 目的 |
|---|---|
| Semantic-only | 不可退化基线 |
| Highlight only | 保留最有潜力 expert |
| Highlight + face | 检验 face 的条件价值 |
| No face-ID | 移除确定无效 expert |
| No face-ID + no scene | 移除当前最严重噪声源 |

虽然去除 scene 会使 oracle ceiling 从约 0.2484 降至 0.2439 MRR，但它可能显著降低 learned selector 的决策难度。应优先追求可学习的稳定收益，而不是保留一个理论 ceiling 高、实际几乎总被误用的 expert。

### 11.3 最小 selector 矩阵

只在 dev 上运行：

```text
semantic-only
marginal_value_greedy, no_face_id_no_scene
set_value_safe, no_face_id
set_value_safe, no_face_id_no_scene
set_value_safe, highlight_only
```

对每个 set-value 配置测试：

```text
min_delta = 0.0 / 0.001 / 0.002 / 0.005
seed = 3 seeds minimum
```

只有同时满足以下条件才进入 10K final：

```text
R@1 delta >= -0.5pp vs semantic-only
MRR delta >= -0.005 vs semantic-only
budget curve maximum degradation <= 0.002
fallback rate and subset regret are stable across seeds
```

### 11.4 改进训练诊断

除 MSE 外必须新增：

- feasible subset top-1 accuracy；
- semantic-vs-expert binary decision accuracy；
- oracle regret；
- predicted gain calibration curve；
- false-positive expert invocation rate；
- 每个 mask 的 precision/recall；
- 按 semantic margin 分桶的实际 delta。

训练目标可进一步考虑：

- subset ranking loss；
- regret-weighted loss；
- 对 empty/semantic-only mask 的显式比较损失；
- 对 rare positive expert cases 的 reweighting；
- expert-specific learned fusion weight。

## 12. 对论文叙事的建议

当前证据支持：

> Conformal safety gate 在 10K gallery 上实现约 77% 的候选缩减，并将 GT false elimination 控制在 5% 目标以内。

当前证据不支持：

> CSER learned selector improves retrieval accuracy over semantic-only or standard baselines.

在 `set_value_safe` 完成且稳定超过 semantic-only 前，建议采用计划中的保守叙事：

```text
Efficient expert routing with conformal safety
```

而不是：

```text
Improved retrieval accuracy through submodular expert routing
```

## 13. 建议研究组重点讨论的问题

1. Scene expert 是修复为 Places365/视频场景模型，还是从当前论文主表移除？
2. 是否接受 Face-ID 在纯文本检索任务中无定义，并将其移出默认 roster？
3. 是否保留 raw score additive fusion，还是改为 learned/calibrated fusion？
4. Selector 的主目标应是 value regression、subset ranking，还是 minimizing oracle regret？
5. 是否将 conformal safety 作为论文主贡献，把 retrieval gain 降为可选经验结果？
6. Final protocol 使用哪个官方 split，如何确保 dev 和 final 完全隔离？
7. 下一轮是否先在 1K/dev 完成全部 gate，再进入 10K？

## 14. 最终判断

本次实验**部分达到预期**：

- Safety、coverage、candidate reduction、cache、诊断文件达到或基本达到预期；
- Legacy selector、budget behavior、retrieval improvement 未达到预期；
- 最终计划的主 selector 和 roster/min-delta 矩阵没有完成；
- 工程和实验协议存在足以影响复现性与结论有效性的缺陷。

本次结果最合理的解释不是“CSER 所有理论均失败”，而是：

> **Legacy marginal-greedy selector 在非单调、专家信号高度不均衡且融合未校准的 retrieval value 上失效；conformal safety 仍然有效；计划中的 SetValueNetwork 尚未得到实验检验。**

在修复 scene/roster、cache-only 路径、版本冻结和 dev/final protocol 之前，不建议再次提交完整 10K 多配置长作业，也不建议将本次结果作为 final paper table。

## 15. 数据与证据来源

主要结果目录：

```text
reports/cser_only/9814419/cser/phase1/
reports/cser_only/9814419/cser/phase2/cser_phase2_marginal_value_all/
reports/cser_only/9814419/cser/phase2/cser_phase2_marginal_density_all/
reports/setup/slurm-cser-msrvtt-updated-9814419.out
reports/setup/slurm-cser-msrvtt-updated-9814419.err
reports/setup/cser_only_latest.status
```

主要代码位置：

```text
cser/data.py
cser/retrieval.py
cser/run_phase2.py
cser/selectors.py
cser/train_svn.py
cser/train_set_value.py
tasks/real_models.py
slurm/run_cser_msrvtt_updated.slurm
```
