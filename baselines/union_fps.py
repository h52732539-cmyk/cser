"""Baseline B: shared uniform-fps sampling (tasks share frames, no stages)."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from core.types import Frame, FrameRequest, SamplingStage, TaskResult
from core.decoder import decode_frames


class UnionFpsBaseline:
    """All tasks share a single uniform-fps decode at max(dense_fps)."""

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

        target_fps = max(
            (t.sub.dense_fps for t in self.tasks.values()), default=1.0,
        )
        stride = 1.0 / max(target_fps, 0.1)
        timestamps = list(np.arange(0.0, max(duration, 1e-6), stride))

        requests: List[FrameRequest] = [
            FrameRequest(
                video_id=video_id,
                frame_idx=int(t * 25),
                timestamp=float(t),
                stage=SamplingStage.DENSE,
                subscribers=set(self.tasks.keys()),
            )
            for t in timestamps
        ]
        frames = decode_frames(video_path, requests)
        self.stats["total_decoded_frames"] = len(frames)

        # For fairness: gated tasks are only run on frames where their
        # gate produced a positive (we emulate by simply passing all frames).
        results: Dict[str, TaskResult] = {}
        for tid, task in self.tasks.items():
            task.reset()
            task.process_dense(frames)
            results[tid] = task.finalize()

        self.stats["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return results
