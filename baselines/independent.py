"""Baseline A: each task samples and decodes independently — worst case."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from core.types import Frame, FrameRequest, SamplingStage, TaskResult
from core.decoder import decode_frames


class IndependentBaseline:
    """Each task samples its own frames at its own `dense_fps`.

    No prefilter, no sharing, no two-stage.
    """

    def __init__(self, tasks) -> None:
        self.tasks = {t.task_id: t for t in tasks}
        self.stats: Dict[str, float] = {}

    def run(
        self,
        video_path: str,
        duration: float,
        video_id: str,
        sensor_stream: Optional[dict] = None,
    ) -> Dict[str, TaskResult]:
        self.stats = {"total_decoded_frames": 0, "total_ms": 0.0}
        t0 = time.perf_counter()
        results: Dict[str, TaskResult] = {}

        total_frames = 0
        for tid, task in self.tasks.items():
            task.reset()
            fps = max(task.sub.dense_fps, task.sub.sparse_fps)
            if fps <= 0.0 or task.sub.gated_by is not None:
                # Gated tasks cannot run in a plain independent baseline
                # (they need detection cues); fall back to their dense_fps.
                fps = max(task.sub.dense_fps, 0.5)
            stride = 1.0 / fps
            timestamps = list(np.arange(0.0, max(duration, 1e-6), stride))
            if len(timestamps) > task.sub.max_frames_dense:
                timestamps = list(np.linspace(
                    0.0, max(duration, 1e-6),
                    task.sub.max_frames_dense, endpoint=False,
                ))
            requests: List[FrameRequest] = [
                FrameRequest(
                    video_id=video_id,
                    frame_idx=int(t * 25),
                    timestamp=float(t),
                    stage=SamplingStage.DENSE,
                    subscribers={tid},
                )
                for t in timestamps
            ]
            frames = decode_frames(video_path, requests)
            total_frames += len(frames)
            task.process_dense(frames)
            results[tid] = task.finalize()

        self.stats["total_decoded_frames"] = total_frames
        self.stats["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return results
