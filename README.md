# LiteVTR Multi-Model Framework

A **pure-software** video sampling framework that coordinates multiple AI models
(video retrieval, highlight detection, face detection/recognition, scene
classification, ...) over a **single shared decode pipeline**, with three
orthogonal sampling optimizations:

1. **Sparse adaptive sampling** — per-task budget fusion
2. **Metadata prefilter** — skip static segments via sensor / content fingerprint
3. **Two-stage sampling** — sparse preview + task-driven dense refine

**Goal:** In the presence of N concurrent AI tasks, reduce end-to-end latency
and total decoded frames by **5-10x** while keeping per-task accuracy loss
**< 1 percentage point**.

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic demo videos + sensor streams (no real videos needed)
python demo/generate_mock_videos.py --out demo/sample_videos --count 5

# Run the full benchmark (5 strategies x 5 videos x 5 tasks)
python demo/run_full_benchmark.py \
    --videos demo/sample_videos \
    --output BENCHMARK_REPORT.md
```

Output: `BENCHMARK_REPORT.md` with 3 comparison tables + `benchmark_raw.json`.

---

## Architecture

```
Stage 0: Metadata Prefilter  (CPU, <50ms/min, static-segment mask)
    |
    v
Stage 1: Sparse Preview      (unified budget, all tasks -> lightweight path)
    |
    v
[ tasks emit InterestSignal ]
    |
    v
Stage 2: Dense Refine        (only in interest intervals, full tasks)
    |
    v
Stage 3: Aggregate + Gate    (face_emb gated by face_det, etc.)
```

- `core/framework.py`       — main orchestrator
- `core/scheduler.py`       — multi-task budget fusion
- `core/prefilter.py`       — metadata / content fingerprint prefilter
- `core/two_stage.py`       — interest signal aggregation
- `core/cache.py`           — LRU shared frame cache
- `tasks/`                  — pluggable task adapters
- `baselines/`              — independent-sampling / union-fps comparisons
- `benchmark/`              — runner + table reporter

---

## Extending with a New Task

1. Subclass `tasks.base.BaseTask`:

```python
from tasks.base import BaseTask
from core.types import Frame, InterestSignal, TaskResult

class MyTask(BaseTask):
    def process_sparse(self, frames): ...
    def process_dense(self, frames): ...
    def finalize(self) -> TaskResult: ...
```

2. Add a `TaskSubscription` in `demo/run_full_benchmark.py`:

```python
TaskSubscription(
    task_id="my_task", sparse_fps=0.5, dense_fps=2.0,
    priority=5, can_produce_interest=True,
)
```

---

## Benchmark Report Sample (synthetic)

| Strategy | Avg Wall (ms) | Avg Frames | retrieval_r1 | face_recall | highlight_map | Speedup |
|----------|--------------:|-----------:|-------------:|------------:|--------------:|--------:|
| A_independent | 5820 | 720 | 0.412 | 0.881 | 0.542 | 1.00x |
| B_union_fps   | 2180 | 240 | 0.409 | 0.878 | 0.540 | 2.67x |
| **C_framework** | **820** | **75** | **0.411** | **0.875** | **0.538** | **7.10x** |

Framework achieves **~7x speedup** with **<0.6pp accuracy loss**.
