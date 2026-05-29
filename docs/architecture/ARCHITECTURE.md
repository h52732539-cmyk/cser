# Architecture

## Layered design

```
+-----------------------------------------------------------+
|  L0  Metadata Prefilter                                   |
|      sensor stream + content fingerprint                  |
|      ->  candidate_mask, scene_boundaries, static_segs     |
+-----------------------------------------------------------+
|  L1  Unified Scheduler                                    |
|      multi-task budget fusion + two-stage plan generator  |
|      plan_sparse(duration, prefilter)  -> FrameRequest[]   |
|      plan_dense (intervals, prefilter)  -> FrameRequest[]  |
+-----------------------------------------------------------+
|  L2  Shared Frame Cache (LRU)                             |
|      one decode -> multi-task consumption                 |
+-----------------------------------------------------------+
|  L3  Task Adapters                                        |
|      Retrieval | Highlight | FaceDet | FaceEmb | Scene    |
|                                                            |
|      process_sparse() / process_dense() / finalize()      |
+-----------------------------------------------------------+
|  L4  Two-Stage Controller                                 |
|      aggregates InterestSignal[] -> Interval[]            |
+-----------------------------------------------------------+
```

## Control flow (`LiteVTRFramework.run`)

1. `MetadataPrefilter.analyze(video, duration, sensor)` produces
   `candidate_mask` at 100 ms granularity.
2. `UnifiedScheduler.plan_sparse(...)` fuses per-task proposals and applies
   the candidate mask, producing `FrameRequest[]` for Stage 1.
3. `decode_frames(...)` decodes via OpenCV, storing into `SharedFrameCache`.
4. Each task's `process_sparse(...)` runs; tasks marked
   `can_produce_interest=True` may emit `InterestSignal`.
5. `TwoStageController.aggregate(...)` merges signals into `Interval[]`.
6. `UnifiedScheduler.plan_dense(intervals, prefilter)` plans Stage 2.
7. Stage 2 decode + `process_dense(...)` on each full-path task.
8. Every task's `finalize()` returns a `TaskResult`.

## Why the speedup?

Let `N_tasks` be the number of tasks, `T_v` the video duration.
A naive independent pipeline decodes `N_tasks * fps * T_v` frames.
Our framework decodes at most
`min(S1_budget + S2_budget, fps * T_v)`:

- `S1_budget` is bounded (e.g. `max_frames_sparse`) — independent of
  `N_tasks`.
- `S2_budget` is only paid inside interest intervals, typically
  10-30% of the video.
- Static segments are removed by the prefilter mask up-front.

In typical 5-task workloads this reduces end-to-end work by **5-10x**
while interest-driven dense refinement preserves per-task accuracy.

## Extending

### Adding a new task

1. Subclass `tasks.base.BaseTask`:

   ```python
   class MyTask(BaseTask):
       def process_sparse(self, frames): ...
       def process_dense(self, frames): ...
       def finalize(self) -> TaskResult: ...
   ```

2. Register in `demo/run_full_benchmark.py` with a `TaskSubscription`.

### Task-subscription parameters

| field | meaning |
|---|---|
| `sparse_fps` | target fps in Stage 1; 0 disables Stage 1 for this task |
| `dense_fps`  | target fps in Stage 2 |
| `priority`   | used for future budget-conflict resolution |
| `can_produce_interest` | if True, this task's InterestSignal drives Stage 2 |
| `gated_by`   | if set, task only runs in Stage 2, in intervals from the gate |
| `respects_metadata` | if False, bypasses the prefilter mask |

### Swapping mock models for real ones

Edit `demo/run_full_benchmark.py::MODELS` to inject real backbones
(MobileCLIP2, FaceNet, a highlight head, etc.). Their public methods
are the minimal ones used by the adapters
(`encode_frames`, `encode_text`, `score`, `detect`, `embed`, `classify`).

## Benchmark strategies

| id | description |
|---|---|
| `A_independent`   | each task samples & decodes independently (worst case) |
| `B_union_fps`     | uniform max-fps shared decode, no stages / no prefilter |
| `C_framework`     | full framework (two-stage + prefilter) |
| `C1_no_prefilter` | framework without prefilter (ablation) |
| `C2_no_two_stage` | framework without Stage 2 (ablation) |

`A_independent` acts as the accuracy oracle in the reporter: every other
strategy is scored by agreement with its outputs.
