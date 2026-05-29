# LiteVTR Multi-Model Framework 项目流程解析

本文档按代码实现梳理本项目的整体流程、核心数学公式、主要模块职责和每个代码文件的作用。

项目本质上不是单一模型训练仓库，而是一个多任务视频检索与视频理解系统框架。它围绕黑盒视觉模型构建工程层优化：减少解码帧数、减少模型调用、复用中间结果，并尽量保持检索和多任务分析精度。

## 1. 总体定位

项目包含两条主线：

1. LiteVTR v1: 多任务共享采样框架
   - 主入口: `core/framework.py`
   - 面向单个视频上的多任务并发执行，例如 retrieval、highlight、face detection、face embedding、scene classification。
   - 重点是共享解码、元数据预过滤、两阶段采样和任务级兴趣区间聚合。

2. LiteVTR++ v2 / C-QIN: 离线索引 + 元数据感知检索 + 查询路由
   - 主入口: `core/framework_v2.py`、`core/offline_index.py`、`routing/qin_model.py`
   - 面向视频库检索。建库时离线编码关键帧，查询时只编码文本，然后用 numpy 检索、元数据过滤和必要时的候选视频精修。
   - C-QIN 用于选择不同检索 route，并通过安全校准确保 hard filter 不容易误删 GT 视频。

## 2. LiteVTR v1 多任务共享采样流程

`LiteVTRFramework.run(video_path, duration, video_id, sensor_stream)` 的执行链路如下：

```text
Stage 0: MetadataPrefilter
    sensor + content fingerprint -> candidate_mask / static_segments / scene_boundaries

Stage 1: Sparse Preview
    UnifiedScheduler -> sparse FrameRequest[]
    decode_frames + SharedFrameCache -> sparse Frame[]
    task.process_sparse() -> InterestSignal[]

Stage 2: Dense Refine
    TwoStageController.aggregate() -> dense intervals
    UnifiedScheduler.plan_dense() -> dense FrameRequest[]
    decode_frames + cache -> dense Frame[]
    task.process_dense()

Stage 3: Finalize
    task.finalize() -> TaskResult
```

### 2.1 Stage 0: 元数据预过滤

实现文件: `core/prefilter.py`

`MetadataPrefilter.analyze()` 生成 100ms 粒度的布尔 mask:

```text
candidate_mask[t] = True  表示该 100ms bucket 可采样
candidate_mask[t] = False 表示可跳过
```

它使用两类信号：

1. Sensor stream
   - gyro 方差低表示静止片段。
   - AF events 可作为 scene boundary。

2. Content fingerprint
   - 以 1fps 粗解码视频。
   - resize 到 64x64。
   - 计算相邻帧平均绝对差。
   - 低差分表示静止，高差分表示场景变化。

若视频时长为 $T$，mask 长度为：

$$
L=\max(1,\lfloor 10T \rfloor)
$$

对于静止区间 $[s,e]$，如果时长超过阈值，则保留第一个 bucket，跳过后续 bucket:

$$
M_{i}=0,\quad i\in[\lfloor 10s \rfloor+1,\lfloor 10e \rfloor)
$$

对 scene boundary $b$，强制保留：

$$
M_{\lfloor 10b \rfloor}=1
$$

### 2.2 Stage 1: 稀疏预览采样

实现文件: `core/scheduler.py`

每个任务有一个 `TaskSubscription`，其中包含：

- `sparse_fps`: Stage 1 采样率
- `dense_fps`: Stage 2 采样率
- `max_frames_sparse`: 稀疏阶段最大帧数
- `max_frames_dense`: 密集阶段最大帧数
- `can_produce_interest`: 是否可以产生兴趣区间
- `gated_by`: 是否只在另一个任务触发的区间里运行
- `respects_metadata`: 是否遵守 metadata prefilter

任务 $i$ 的稀疏采样提议为：

$$
P_i^s=\left\{\frac{k}{f_i^s}\mid 0\le \frac{k}{f_i^s}<T\right\}
$$

若超出预算 $B_i^s$，代码用 `np.linspace` 截断：

$$
|P_i^s|\le B_i^s
$$

所有任务提议合并为：

$$
P^s=\operatorname{MergeGap}\left(\bigcup_i P_i^s,g\right)
$$

其中 $g$ 是 `merge_gap_sec`，默认 0.05 秒。

再应用预过滤：

$$
\tilde P^s=\{t\in P^s\mid M_{\lfloor 10t \rfloor}=1\}
$$

最终生成 `FrameRequest`，其中 `subscribers` 记录哪些任务消费该帧。

### 2.3 解码与帧缓存

实现文件:

- `core/decoder.py`
- `core/cache.py`

`decode_frames()` 逐个处理 `FrameRequest`：

1. 先查 `SharedFrameCache`。
2. 命中则复用 RGB frame。
3. 未命中则用 OpenCV seek 到 `timestamp` 解码。
4. 解码后写入 LRU 缓存。

这样 Stage 1 和 Stage 2 若请求同一帧，可以避免重复解码。

### 2.4 Stage 1 任务执行与兴趣区间

实现文件:

- `tasks/retrieval_task.py`
- `tasks/highlight_task.py`
- `tasks/face_task.py`
- `tasks/scene_task.py`

每个任务实现统一接口：

```python
process_sparse(frames) -> Optional[InterestSignal]
process_dense(frames) -> None
finalize() -> TaskResult
```

典型兴趣区间形式为：

$$
I_j=[\max(0,t_j-w),t_j+w]
$$

其中 $t_j$ 是高分帧时间戳，$w$ 是任务窗口半径。

RetrievalTask 使用 CLIP frame embedding 与 query embedding 的余弦分数：

$$
s(t,q)=e_t^\top q
$$

若分数高于阈值，则产生兴趣区间。

HighlightTask 使用 highlight model 的 frame score。

FaceDetectionTask 使用人脸置信度，检测到人脸后产生兴趣区间，用于驱动 FaceEmbeddingTask。

SceneClassificationTask 默认不产生兴趣区间，主要做稀疏分类。

### 2.5 Stage 2 兴趣区间聚合与密集采样

实现文件: `core/two_stage.py`

所有任务的 `InterestSignal` 会被展开、合并、排序和截断。

先扩展每个区间：

$$
I'_j=[\max(0,start_j-\epsilon),end_j+\epsilon]
$$

若两个区间间隔小于 `merge_gap_sec`，则合并：

$$
[a,b]\cup[c,d]=[a,\max(b,d)] \quad \text{if } c-b\le g
$$

合并后的分数取最大值：

$$
score=\max(score_1,score_2)
$$

再按 score 降序取前 `max_intervals` 个，并限制总时长：

$$
\sum_j |I_j|\le B_{\text{time}}
$$

密集采样由 `UnifiedScheduler.plan_dense()` 完成。对任务 $i$ 和兴趣区间集合 $\mathcal{I}$：

$$
P_i^d=\bigcup_{I=[a,b]\in\mathcal{I}}\left\{a+\frac{k}{f_i^d}\mid a+\frac{k}{f_i^d}<b\right\}
$$

最终同样进行近邻合并、metadata mask 过滤和订阅者合并。

### 2.6 结果汇总与时间片段生成

实现文件: `core/segment_aggregator.py`

多个任务会输出帧级分数序列：

$$
\{(t_i,s_i)\}_{i=1}^n
$$

`SegmentAggregator` 的流程是：

1. 按时间排序。
2. 对分数做 moving average 平滑。
3. 用分位数或绝对阈值选高分点。
4. 连续高分点组成 raw segments。
5. 合并间隔很近的片段。
6. 删除过短片段。
7. temporal NMS。
8. 按 score 取 top-K。

平滑公式：

$$
\tilde s_i=\frac{1}{w}\sum_{j=i-r}^{i+r}s_j
$$

分位数阈值：

$$
\theta=Q_p(\tilde s)
$$

保留满足：

$$
\tilde s_i\ge \theta
$$

Temporal IoU:

$$
\operatorname{IoU}(A,B)=
\frac{|A\cap B|}{|A\cup B|}
$$

NMS 中如果某候选片段与已保留片段 IoU 超过阈值，则丢弃。

## 3. v1 与 Baseline 的效率对比

Baseline A: Independent

实现文件: `baselines/independent.py`

每个任务独立采样、独立解码：

$$
F_{\text{independent}}=\sum_i \min(f_iT,B_i)
$$

Baseline B: Union FPS

实现文件: `baselines/union_fps.py`

所有任务共享一个最高 fps 的均匀采样：

$$
F_{\text{union}}=\max_i(f_i)T
$$

LiteVTR:

$$
F_{\text{LiteVTR}}=|\tilde P^s|+|\tilde P^d|
$$

其中 $\tilde P^d$ 只在兴趣区间内产生。若兴趣区间总长度为 $\rho T$，$\rho\ll 1$，则可近似认为：

$$
F_{\text{LiteVTR}}\approx B_s+\rho T f_d
$$

相对独立采样的速度收益：

$$
\operatorname{Speedup}\approx
\frac{\sum_i f_iT}{B_s+\rho Tf_d}
$$

这也是 README 中宣称 5-10x 加速的主要来源。

## 4. LiteVTR++ v2 离线索引检索流程

实现文件:

- `core/framework_v2.py`
- `core/offline_index.py`
- `core/query_planner.py`
- `core/adaptive_sampler.py`
- `core/cross_task_cache.py`

v2 的目标是在不修改黑盒视觉模型权重的情况下，把昂贵图像塔调用尽可能前移到离线阶段。

### 4.1 离线建库

`OfflineIndexBuilder.build_one()` 对每个视频做：

1. 用 sampler 选关键帧时间戳。
2. 解码这些关键帧。
3. 调用 image encoder 得到帧向量。
4. L2 归一化帧向量。
5. 构建多 K prototype。
6. 可选建立 face timeline、scene timeline、metadata。

给定帧向量：

$$
E_v=[e_1,e_2,\dots,e_n],\quad e_i\in\mathbb{R}^D
$$

每个向量归一化：

$$
\hat e_i=\frac{e_i}{\|e_i\|_2+\epsilon}
$$

对 $K$ 个等长时间段构建 prototype：

$$
p_k=\frac{1}{|S_k|}\sum_{i\in S_k}\hat e_i
$$

再归一化：

$$
\hat p_k=\frac{p_k}{\|p_k\|_2+\epsilon}
$$

默认 `k_values=(2,4,6)`，即一个视频会保留多粒度表示。

### 4.2 查询时基础检索

查询文本先编码并归一化：

$$
q=\frac{E_{\text{text}}(query)}{\|E_{\text{text}}(query)\|_2+\epsilon}
$$

基础视频分数使用最细粒度 K 的 prototype 最大余弦：

$$
s_{\text{base}}(v,q)=\max_{p\in P_v^{K_{\max}}}q^\top p
$$

### 4.3 QAMP 分数

QAMP 对一个视频内部的 prototype 做 soft attention。

对视频 $v$ 的 prototypes $\{p_j\}$，先算：

$$
a_j=q^\top p_j
$$

softmax 权重为：

$$
w_j=\frac{\exp(a_j/\tau)}{\sum_l \exp(a_l/\tau)}
$$

QAMP 分数：

$$
s_{\text{qamp}}(v,q)=\sum_jw_ja_j
$$

多 K 时：

$$
s_{\text{qamp}}(v,q)=\frac{1}{|\mathcal{K}|}\sum_{K\in\mathcal{K}}s_{\text{qamp}}^{K}(v,q)
$$

### 4.4 NNN 校正

NNN 用于缓解某些视频对很多 query 都高分的 hubness 问题。

$$
s_{\text{nnn}}(v,q)=\frac{s_{\text{base}}(v,q)-\mu_v}{\sigma_v+\epsilon}
$$

batch 检索时，$\mu_v,\sigma_v$ 来自所有 query 对该视频的分数列：

$$
\mu_v=\frac{1}{N_q}\sum_qs_{\text{base}}(v,q)
$$

$$
\sigma_v=\sqrt{\frac{1}{N_q}\sum_q(s_{\text{base}}(v,q)-\mu_v)^2}
$$

### 4.5 融合与 col-softmax

对 top-M 候选，代码将 NNN 和 QAMP 分别 z-score，再融合：

$$
z_{\text{nnn}}=\frac{s_{\text{nnn}}-\operatorname{mean}(s_{\text{nnn}})}
{\operatorname{std}(s_{\text{nnn}})+\epsilon}
$$

$$
z_{\text{qamp}}=\frac{s_{\text{qamp}}-\operatorname{mean}(s_{\text{qamp}})}
{\operatorname{std}(s_{\text{qamp}})+\epsilon}
$$

$$
s_{\text{fused}}=(1-\alpha)z_{\text{nnn}}+\alpha z_{\text{qamp}}
$$

batch 路径中，col-softmax 沿 query 维度对每个视频做归一化：

$$
s'_{q,v}=
\frac{\exp(s_{q,v}/\beta)}
{\sum_{q'}\exp(s_{q',v}/\beta)}
$$

单 query 路径中，因为没有 query batch，代码近似为 across videos softmax。

### 4.6 v2 查询规划

实现文件: `core/query_planner.py`

检索返回 top hits 后，使用 top1-top2 margin 判断查询难度：

$$
margin=s_1-s_2
$$

决策规则：

$$
margin\ge\tau_{\text{easy}}\Rightarrow EASY
$$

$$
\tau_{\text{hard}}\le margin<\tau_{\text{easy}}\Rightarrow MEDIUM
$$

$$
margin<\tau_{\text{hard}}\Rightarrow HARD
$$

EASY 直接返回离线索引结果。MEDIUM 对 top-3 视频精修。HARD 对 top-10 视频精修，并可启用 MomentDETR/highlight。

### 4.7 候选视频精修

`LiteVTRFrameworkV2._refine_one_video()` 对单个候选视频做：

1. MetadataPrefilter。
2. HybridSampler 采样。
3. 应用 candidate mask。
4. 解码采样帧。
5. 生成 `FrameIdentity`。
6. 通过 `CrossTaskCache` 复用模型输出。
7. 调用图像编码器得到 frame embeddings。
8. 与 query embedding 做 cosine。
9. 用 `SegmentAggregator` 输出时间片段。
10. 可选调用 highlight/MomentDETR 辅助生成片段。

精修分数取帧级最大相似度：

$$
s_{\text{refine}}(v,q)=\max_t e_t^\top q
$$

最终与离线索引分数混合：

$$
s_{\text{final}}=0.7s_{\text{index}}+0.3s_{\text{refine}}
$$

## 5. 元数据感知检索

相关文件:

- `core/query_parser.py`
- `core/metadata.py`
- `core/meta_filter.py`
- `metadata/noisy_metadata.py`

### 5.1 Query intent

自然语言 query 被解析为：

$$
Intent=(semantic\_text,time,geo,motion,device)
$$

其中：

- `time_window`: POSIX timestamp 区间
- `geo_categories`: coast、mountain、urban 等
- `motion_classes`: running、walking、vehicle 等
- `device_filter`: huawei、iphone、samsung 等

### 5.2 硬过滤

对每个视频 $v$，各元数据轴生成 bool mask：

$$
M_a(v)\in\{0,1\}
$$

多个约束取交集：

$$
M(v)=\bigwedge_{a\in A}M_a(v)
$$

若 metadata 缺失，默认非 strict 模式下保留该视频，避免误删。

### 5.3 软分数

软元数据分数按轴计算。

时间轴使用窗口内满分、slack 区间线性衰减：

$$
s_{\text{time}}(v)=
\begin{cases}
1, & |t-c|\le h\\
0, & |t-c|\ge h+\Delta\\
1-\frac{|t-c|-h}{\Delta}, & \text{otherwise}
\end{cases}
$$

其中 $c$ 是 query 时间窗口中心，$h$ 是半窗口长度，$\Delta$ 是 slack。

多轴组合时：

$$
s_{\text{meta}}(v)=
\left(\prod_{a\in A}s_a(v)\right)^{1/|A|}
$$

语义和元数据融合：

$$
s(v)=\alpha s_{\text{sem}}(v)+(1-\alpha)s_{\text{meta}}(v)
$$

代码默认关闭软融合，更偏向 hard filter 后语义排序，因为粗粒度 metadata 软分数可能抬高大量同类但不相关的视频。

## 6. C-QIN 路由系统

相关文件:

- `routing/route_schema.py`
- `routing/route_bank.py`
- `routing/route_executor.py`
- `routing/route_bank_builder.py`
- `routing/qin_model.py`
- `routing/train_qin.py`
- `routing/calibrate_safety.py`
- `routing/calibrated_planner.py`
- `routing/calibrated_planner_v2.py`

### 6.1 Route 定义

`RetrievalRoute` 描述一次检索策略：

- `hard_axes`: 哪些 metadata 轴用于硬过滤
- `soft_axes`: 哪些 metadata 轴用于软排序
- `candidate_topm`: rerank 的候选数量
- `rerank_mode`: none、qamp、nnn_qamp、col_softmax_post_filter
- `budget_tier`: low、medium、high、full
- `allow_image_model_calls`: 是否允许图像模型调用
- `allow_dense_refinement`: 是否允许 dense refinement

`configs/route_bank_30.yaml` 定义约 30 条 route，包括纯语义、单轴 hard filter、多轴 hard filter、soft rerank 和 full budget dense refine。

### 6.2 RouteExecutor

`RouteExecutor.execute()` 流程：

1. 调用 OfflineIndex 得到语义分数。
2. 根据 route 的 hard axes 构造 filter intent。
3. 应用 hard filter。
4. 如果 route 要求 `col_softmax_post_filter`，则在过滤后的候选集合上做 softmax。
5. 根据 soft axes 做软融合。
6. 计算 GT 视频 rank、recall@1/5/10、gt_filtered、candidate_count、cost_proxy。

### 6.3 反事实 route 标签

`route_bank_builder.py` 对每个 query 和每条 route 都执行一次，生成训练标签。

utility 定义为：

$$
U=
\begin{cases}
-\lambda_f, & \text{GT 被过滤或 rank < 0}\\
\frac{1}{rank+1}+0.5\mathbf{1}[rank=0]+0.1\mathbf{1}[rank<5]-0.05cost,
& \text{otherwise}
\end{cases}
$$

每个 query 的 oracle route:

$$
r^*=\arg\max_r U(q,r)
$$

### 6.4 C-QIN 输入特征

实现文件: `routing/qin_model.py`

特征大约 531 维：

$$
x=[
q_{\text{clip}}^{512},
qpp^{6},
keyword^{5},
metaAvailability^{4},
budget^{4}
]
$$

其中 QPP 包括 top1、top2、margin、entropy、std、concentration。

### 6.5 C-QIN 模型

模型是一个小型 MLP，双头输出：

$$
h=\operatorname{MLP}(x)
$$

Route value head:

$$
\hat u=W_uh+b_u
$$

Safety head:

$$
\hat z=W_sh+b_s
$$

$$
\hat p=\sigma(\hat z)
$$

$\hat p$ 是 time、geo、motion、device 四个轴的 GT survival probability。

### 6.6 训练损失

实现文件: `routing/train_qin.py`

先对每个 query 的 route utility 归一化：

$$
u'_{q,r}=\frac{u_{q,r}-\min_r u_{q,r}}{\max_r u_{q,r}-\min_r u_{q,r}+\epsilon}
$$

总损失：

$$
\mathcal{L}
=
\operatorname{Huber}(\hat u,u')
+\alpha\operatorname{CE}(\hat u,r^*)
+\beta\operatorname{BCEWithLogits}(\hat z,y_{\text{safety}})
$$

其中：

- Huber 学每条 route 的 utility。
- CE 学 oracle route。
- BCE 学每个 metadata 轴是否安全。

### 6.7 安全校准

实现文件: `routing/calibrate_safety.py`

对每个 metadata 轴选择阈值 $\tau$，使得当模型预测安全概率高于 $\tau$ 时，误过滤 GT 的概率被控制。

若校准集上 accepted 样本数为 $n$，失败数为 $k$，选择最低的 $\tau$，满足：

$$
UCB_{\text{CP}}(k,n,\alpha)\le \delta
$$

其中 $UCB_{\text{CP}}$ 是 Clopper-Pearson binomial failure rate 上置信界。

### 6.8 单阈值 planner

实现文件: `routing/calibrated_planner.py`

对每条 route，如果所有 hard axes 都满足：

$$
p_a\ge \tau_a
$$

则该 route 安全。Planner 在安全 route 中选择：

$$
r=\arg\max_{r\in\mathcal{R}_{safe}}\hat u_r
$$

若无安全 route，则 fallback 到纯语义 route。

### 6.9 双阈值 planner v2

实现文件: `routing/calibrated_planner_v2.py`

每个轴分为三种 zone：

$$
p_a\ge\tau_{\text{hard}}\Rightarrow hard\_safe
$$

$$
\tau_{\text{soft}}\le p_a<\tau_{\text{hard}}\Rightarrow soft\_uncertain
$$

$$
p_a<\tau_{\text{soft}}\Rightarrow unsafe
$$

这样可以避免单阈值版本中只要某个轴不安全就完全退回纯语义的问题。v2 会把 uncertain 轴降级为 soft rerank。

`BudgetedCascadePlanner` 还会在低预算 route 效果差时升级到更高预算 route。

## 7. 文件职责总览

### 7.1 core

| 文件 | 作用 |
|---|---|
| `core/types.py` | 定义 `Frame`、`Interval`、`InterestSignal`、`TaskResult`、`FrameRequest` 等核心数据结构 |
| `core/subscription.py` | 定义任务订阅参数，例如 sparse/dense fps、预算、是否产出兴趣区、是否 gated |
| `core/framework.py` | v1 主编排器，串联预过滤、稀疏采样、任务执行、密集精修和结果汇总 |
| `core/framework_v2.py` | v2 查询编排器，串联文本编码、离线检索、查询规划、候选精修 |
| `core/scheduler.py` | 多任务采样计划融合，生成共享 `FrameRequest` |
| `core/prefilter.py` | 基于 sensor 和内容指纹生成候选 mask、静止片段和场景边界 |
| `core/two_stage.py` | 聚合 Stage 1 兴趣信号，得到 Stage 2 密集采样区间 |
| `core/decoder.py` | OpenCV 解码指定时间戳帧，并提供视频探测函数 |
| `core/cache.py` | 解码帧 LRU 缓存 |
| `core/cross_task_cache.py` | 跨任务和跨模型输出缓存，按帧身份复用模型结果 |
| `core/frame_identity.py` | 帧身份构造，包括 byte hash、pHash、embedding cosine |
| `core/adaptive_sampler.py` | Uniform、内容差分、运动向量、Q-Frame、Hybrid 五类采样器 |
| `core/segment_aggregator.py` | 将帧级分数转换为时间片段 |
| `core/offline_index.py` | 离线视频索引、prototype 构建、QAMP/NNN/col-softmax 检索 |
| `core/query_parser.py` | 规则式 query 意图解析 |
| `core/meta_filter.py` | 元数据硬过滤、软打分和语义元数据融合 |
| `core/metadata.py` | 视频元数据结构、ffprobe 提取、IMU 运动分类、粗地理分类 |
| `core/query_planner.py` | 基于 margin 的 EASY/MEDIUM/HARD 查询规划 |
| `core/__init__.py` | core 包导出 |

### 7.2 tasks

| 文件 | 作用 |
|---|---|
| `tasks/base.py` | 所有任务适配器抽象基类 |
| `tasks/retrieval_task.py` | CLIP 检索任务，产生高相似度兴趣区和最终检索片段 |
| `tasks/highlight_task.py` | highlight 分数任务，输出精彩片段 |
| `tasks/face_task.py` | 人脸检测任务和被人脸检测 gated 的人脸 embedding 任务 |
| `tasks/scene_task.py` | 场景分类任务，输出标签序列和 dominant scene |
| `tasks/mock_models.py` | 确定性 mock 模型，便于无权重 benchmark |
| `tasks/real_models.py` | 真实模型适配器，包括 CLIP、MomentDETR、InsightFace、MobileNetV3 等 |
| `tasks/__init__.py` | tasks 包导出 |

### 7.3 routing

| 文件 | 作用 |
|---|---|
| `routing/route_schema.py` | 定义 `RetrievalRoute` 的结构和校验规则 |
| `routing/route_bank.py` | 从 YAML 加载和管理 route bank |
| `routing/route_executor.py` | 执行单条 route，返回 rank、GT 是否被过滤、cost、recall |
| `routing/route_bank_builder.py` | 对 query-route 组合生成反事实训练标签 |
| `routing/qin_model.py` | C-QIN 双头网络和特征提取 |
| `routing/train_qin.py` | C-QIN 训练循环 |
| `routing/calibrate_safety.py` | 安全阈值校准 |
| `routing/calibrated_planner.py` | 单阈值安全路由 planner |
| `routing/calibrated_planner_v2.py` | 双阈值 planner，支持 hard/soft/unsafe 分区和预算级联 |
| `routing/baselines.py` | AAAI 对比基线 B0-B8/B9/B10 |
| `routing/__init__.py` | routing 包初始化 |

### 7.4 baselines

| 文件 | 作用 |
|---|---|
| `baselines/independent.py` | Baseline A，每个任务独立采样和解码 |
| `baselines/union_fps.py` | Baseline B，所有任务共享最高 fps 的均匀采样 |
| `baselines/__init__.py` | baselines 包导出 |

### 7.5 benchmark 与 eval

| 文件 | 作用 |
|---|---|
| `benchmark/runner.py` | 运行 A/B/C/C1/C2 策略并收集统计 |
| `benchmark/metrics.py` | 与 oracle baseline 比较任务准确性 |
| `benchmark/reporter.py` | 生成 Markdown benchmark 表格 |
| `benchmark/__init__.py` | benchmark 包导出 |
| `eval/metrics.py` | 检索、safety、cost、oracle gap 指标 |
| `eval/eval_planner.py` | 统一评估 planner 或 baseline |
| `eval/__init__.py` | eval 包初始化 |

### 7.6 demo

| 文件 | 作用 |
|---|---|
| `demo/generate_mock_videos.py` | 生成合成视频和 sensor sidecar |
| `demo/run_full_benchmark.py` | v1 完整多任务 benchmark |
| `demo/run_benchmark_v2.py` | v2 offline/index benchmark |
| `demo/run_msrvtt_v2.py` | MSR-VTT 离线索引检索评估 |
| `demo/run_msrvtt_meta_v3.py` | MSR-VTT 合成元数据检索评估 |
| `demo/run_msrvtt_regression.py` | 真实 MSR-VTT 多任务回归 |
| `demo/run_multitask_regression.py` | v2 多任务准确性回归 |
| `demo/run_multitask_regression_v2.py` | 更严格的黑盒一致性回归 |
| `demo/sanity_msrvtt.py` | MSR-VTT cache 纯 cosine sanity check |

### 7.7 experiments

| 文件 | 作用 |
|---|---|
| `experiments/ablation.py` | LiteVTR++ 模块消融框架 |
| `experiments/run_ablation_msrvtt.py` | MSR-VTT 完整模块消融 |
| `experiments/run_retrieval_ablation.py` | 检索打分模块消融 |
| `experiments/run_sampling_ablation.py` | 采样、解码、模型调用消融 |
| `experiments/run_qvh_eval.py` | QVHighlights moment localization 评估 |

### 7.8 scripts

| 文件 | 作用 |
|---|---|
| `scripts/run_cqin_pipeline.py` | C-QIN 端到端管线: 建标签、训练、校准、评估 |
| `scripts/run_cqin_pipeline_v2.py` | C-QIN v2: 双阈值、软 fallback、显著性检验 |
| `scripts/run_final_eval.py` | AAAI final evaluation 主脚本 |
| `scripts/run_cross_dataset.py` | MSR-VTT 训练、QVHighlights 零样本迁移 |
| `scripts/run_pareto_sweep.py` | 成本和精度 Pareto 曲线 |
| `scripts/run_route_bank_ablation.py` | 不同 route bank 组成的消融 |
| `scripts/run_calibration_sweep.py` | 安全阈值和 soft ratio 网格扫描 |
| `scripts/run_training_sensitivity.py` | C-QIN 训练数据量敏感性 |
| `scripts/run_head_analysis.py` | C-QIN head 校准、冲突、特征重要性分析 |
| `scripts/run_moment_localization.py` | 检索 top-K 后做 moment localization |
| `scripts/gen_huawei_docx.py` | 生成交付用 Word 文档 |

### 7.9 metadata

| 文件 | 作用 |
|---|---|
| `metadata/noisy_metadata.py` | 给合成元数据注入缺失、偏移、错误标签噪声 |
| `metadata/__init__.py` | metadata 包初始化 |

### 7.10 tests

| 文件 | 作用 |
|---|---|
| `tests/test_core.py` | v1 scheduler、two-stage、cache、segment 测试 |
| `tests/test_v2.py` | v2 frame identity、cross cache、sampler、offline index、planner 测试 |
| `tests/test_meta.py` | 元数据解析、过滤、融合测试 |
| `tests/eval/test_metrics.py` | eval 指标测试 |
| `tests/routing/test_route_schema.py` | route schema 测试 |
| `tests/routing/test_route_bank.py` | route bank 测试 |
| `tests/routing/test_route_executor.py` | route executor 测试 |
| `tests/routing/test_calibration.py` | safety calibration 和 planner 测试 |
| `tests/routing/test_qin_model.py` | QIN model 和训练 smoke test |

## 8. 关键结论

本项目的核心思路可以概括为：

1. 对单视频多任务，使用共享解码和两阶段采样，将多任务重复解码变成一次共享采样。
2. 对视频库检索，使用离线关键帧索引，把图像塔调用从查询时转移到建库时。
3. 对带时间、地点、运动、设备约束的查询，用 metadata hard filter 缩小候选集。
4. 用 QAMP、NNN、col-softmax 改善纯 cosine 检索排序。
5. 用 C-QIN 学习 query 到 retrieval route 的选择，并用安全校准确保 hard filter 风险可控。
6. 用 cross-task cache 和 frame identity 保证相同或近似帧的模型输出可复用。

一句话总结：LiteVTR 的主要贡献不是训练更大的视频模型，而是在黑盒模型外构建一层高效、可校准、可消融的视频检索和多任务执行系统。
