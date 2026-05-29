"""Highlight detection task."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.segment_aggregator import SegmentAggregator
from core.types import Frame, InterestSignal, Interval, TaskResult
from .base import BaseTask


class HighlightTask(BaseTask):
    """Per-frame highlight score + SegmentAggregator for [t_start, t_end]."""

    def __init__(
        self,
        subscription,
        model,
        interest_threshold: float = 0.60,
        interest_window_sec: float = 1.5,
        # Segment aggregation
        seg_percentile: float = 0.75,
        seg_smooth_window: int = 3,
        seg_merge_gap_sec: float = 1.0,
        seg_min_length_sec: float = 0.3,
        seg_max_per_query: int = 10,
        seg_boundary_pad_sec: float = 0.0,
    ) -> None:
        super().__init__(subscription)
        self.model = model
        self.th_sparse = interest_threshold
        self.win = interest_window_sec
        self.aggregator = SegmentAggregator(
            percentile=seg_percentile,
            smooth_window=seg_smooth_window,
            merge_gap_sec=seg_merge_gap_sec,
            min_segment_sec=seg_min_length_sec,
            max_segments=seg_max_per_query,
            pad_sec=seg_boundary_pad_sec,
        )
        self.reset()

    def reset(self) -> None:
        self._scores: List[tuple] = []  # (timestamp, score)

    def process_sparse(self, frames: List[Frame]) -> Optional[InterestSignal]:
        if not frames:
            return InterestSignal(self.task_id, [])
        images = [f.image for f in frames]
        scores = self.model.score(images)
        intervals: List[Interval] = []
        for f, s in zip(frames, scores):
            self._scores.append((float(f.timestamp), float(s)))
            if s > self.th_sparse:
                intervals.append(Interval(
                    start=max(0.0, f.timestamp - self.win),
                    end=f.timestamp + self.win,
                    score=float(s),
                    source_task=self.task_id,
                ))
        return InterestSignal(self.task_id, intervals)

    def process_dense(self, frames: List[Frame]) -> None:
        if not frames:
            return
        images = [f.image for f in frames]
        scores = self.model.score(images)
        for f, s in zip(frames, scores):
            self._scores.append((float(f.timestamp), float(s)))

    def finalize(self) -> TaskResult:
        segs = self.aggregator.aggregate(self._scores)

        payload = {
            "segments": [s.to_dict() for s in segs],
            "n_scored_frames": len(self._scores),
        }
        map_like = (
            np.mean([s.score for s in segs]) if segs else 0.0
        )
        return TaskResult(
            task_id=self.task_id,
            payload=payload,
            metrics={
                "map_like": float(map_like),
                "n_segments": float(len(segs)),
            },
        )
