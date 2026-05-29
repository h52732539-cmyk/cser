"""Face detection + face embedding tasks (face embedding gated by detection)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.segment_aggregator import SegmentAggregator
from core.types import Frame, InterestSignal, Interval, TaskResult
from .base import BaseTask


class FaceDetectionTask(BaseTask):
    """Binary face-present detection per frame.

    Also publishes face-present intervals as InterestSignal so that
    dense sampling covers regions where faces exist (giving FaceEmbeddingTask
    more frames to embed).
    """

    def __init__(
        self,
        subscription,
        detector,
        interest_window_sec: float = 1.0,
        det_threshold: float = 0.5,
        # Segment aggregation for face-present intervals
        seg_percentile: float = 0.50,
        seg_smooth_window: int = 3,
        seg_merge_gap_sec: float = 1.5,
        seg_min_length_sec: float = 0.2,
        seg_max_per_query: int = 10,
    ) -> None:
        super().__init__(subscription)
        self.detector = detector
        self.win = interest_window_sec
        self.det_threshold = det_threshold
        self.aggregator = SegmentAggregator(
            percentile=seg_percentile,
            smooth_window=seg_smooth_window,
            merge_gap_sec=seg_merge_gap_sec,
            min_segment_sec=seg_min_length_sec,
            max_segments=seg_max_per_query,
        )
        self.reset()

    def reset(self) -> None:
        self._detections: List[Dict] = []
        # shared with FaceEmbeddingTask via framework: we expose the list
        # through a simple dict shared state is NOT required here, because
        # the gated task receives frames only in intervals emitted by this
        # task's InterestSignal.

    def process_sparse(self, frames: List[Frame]) -> Optional[InterestSignal]:
        if not frames:
            return InterestSignal(self.task_id, [])
        images = [f.image for f in frames]
        dets = self.detector.detect(images)
        intervals: List[Interval] = []
        for f, (present, conf) in zip(frames, dets):
            self._detections.append({
                "timestamp": float(f.timestamp),
                "present": bool(present),
                "conf": float(conf),
                "stage": f.stage.value,
            })
            if present and conf >= self.det_threshold:
                intervals.append(Interval(
                    start=max(0.0, f.timestamp - self.win),
                    end=f.timestamp + self.win,
                    score=float(conf),
                    source_task=self.task_id,
                ))
        return InterestSignal(self.task_id, intervals)

    def process_dense(self, frames: List[Frame]) -> None:
        if not frames:
            return
        images = [f.image for f in frames]
        dets = self.detector.detect(images)
        for f, (present, conf) in zip(frames, dets):
            self._detections.append({
                "timestamp": float(f.timestamp),
                "present": bool(present),
                "conf": float(conf),
                "stage": f.stage.value,
            })

    def finalize(self) -> TaskResult:
        n_total = len(self._detections)
        n_pos = sum(1 for d in self._detections if d["present"])
        recall = (n_pos / n_total) if n_total > 0 else 0.0

        # Build face-present segments from (timestamp, confidence) pairs.
        pairs = [
            (d["timestamp"], d["conf"] if d["present"] else 0.0)
            for d in self._detections
        ]
        segs = self.aggregator.aggregate(pairs) if pairs else []

        return TaskResult(
            task_id=self.task_id,
            payload={
                "detections": sorted(self._detections,
                                      key=lambda d: d["timestamp"]),
                "segments": [s.to_dict() for s in segs],
                "n_positive": n_pos,
                "n_total": n_total,
            },
            metrics={
                "det_recall": float(recall),
                "n_positive": float(n_pos),
                "n_segments": float(len(segs)),
            },
        )


# ----------------------------------------------------------------------


class FaceEmbeddingTask(BaseTask):
    """Face embedding, gated by FaceDetectionTask.

    This task has `sparse_fps=0` so it contributes nothing to Stage 1.
    It runs only in Stage 2 on frames in face-present intervals.
    """

    def __init__(self, subscription, embedder) -> None:
        super().__init__(subscription)
        self.embedder = embedder
        self.reset()

    def reset(self) -> None:
        self._embs: List[np.ndarray] = []
        self._ts: List[float] = []

    def process_sparse(self, frames: List[Frame]) -> Optional[InterestSignal]:
        # Gated task: framework skips Stage 1, but we implement for safety.
        return None

    def process_dense(self, frames: List[Frame]) -> None:
        if not frames:
            return
        images = [f.image for f in frames]
        embs = self.embedder.embed(images)
        self._embs.extend(embs)
        self._ts.extend(f.timestamp for f in frames)

    def finalize(self) -> TaskResult:
        embs_list = [e.tolist() for e in self._embs]
        return TaskResult(
            task_id=self.task_id,
            payload={
                "embeddings": embs_list,
                "timestamps": [float(t) for t in self._ts],
                "n_embeddings": len(self._embs),
            },
            metrics={"n_embeddings": float(len(self._embs))},
        )
