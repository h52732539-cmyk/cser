"""Video retrieval task (LiteVTR-style) with [t_start, t_end] segment output."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.segment_aggregator import SegmentAggregator
from core.types import Frame, InterestSignal, Interval, TaskResult
from .base import BaseTask


class RetrievalTask(BaseTask):
    """Text-to-video retrieval with segment-level localization.

    Stage 1: encode frames with the CLIP image tower, compute cosine vs
             each query, emit InterestSignal over high-score regions.
    Stage 2: densely encode frames inside interest regions.
    Finalize:
        - frame-level top-K timestamps (legacy output)
        - [t_start, t_end] segment ranking per query (new)
    """

    def __init__(
        self,
        subscription,
        clip_model,
        query_embeddings: List[np.ndarray],
        top_k: int = 5,
        interest_score_threshold: float = 0.05,
        interest_window_sec: float = 1.5,
        # Segment aggregation
        seg_percentile: float = 0.80,
        seg_smooth_window: int = 3,
        seg_merge_gap_sec: float = 0.8,
        seg_min_length_sec: float = 0.3,
        seg_max_per_query: int = 5,
        seg_boundary_pad_sec: float = 0.0,
    ) -> None:
        super().__init__(subscription)
        self.clip = clip_model
        self.queries = [np.asarray(q) for q in query_embeddings]
        self.top_k = top_k
        self.thr = interest_score_threshold
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
        self._sparse_embs: List[np.ndarray] = []
        self._sparse_ts: List[float] = []
        self._dense_embs: List[np.ndarray] = []
        self._dense_ts: List[float] = []

    # ------------------------------------------------------------------

    def process_sparse(self, frames: List[Frame]) -> Optional[InterestSignal]:
        if not frames:
            return InterestSignal(self.task_id, [])
        images = [f.image for f in frames]
        embs = self.clip.encode_frames(images)
        self._sparse_embs.extend(embs)
        self._sparse_ts.extend(f.timestamp for f in frames)

        embs_np = np.stack(embs)
        intervals: List[Interval] = []
        for q in self.queries:
            sims = embs_np @ q
            top_idx = np.argsort(-sims)[:3]
            for i in top_idx:
                if sims[i] > self.thr:
                    t = frames[i].timestamp
                    intervals.append(Interval(
                        start=max(0.0, t - self.win),
                        end=t + self.win,
                        score=float(sims[i]),
                        source_task=self.task_id,
                    ))
        return InterestSignal(self.task_id, intervals)

    def process_dense(self, frames: List[Frame]) -> None:
        if not frames:
            return
        images = [f.image for f in frames]
        embs = self.clip.encode_frames(images)
        self._dense_embs.extend(embs)
        self._dense_ts.extend(f.timestamp for f in frames)

    def finalize(self) -> TaskResult:
        embs = self._sparse_embs + self._dense_embs
        ts = self._sparse_ts + self._dense_ts

        top_k_per_query: List[List[dict]] = []
        segments_per_query: List[List[dict]] = []

        if embs:
            m = np.stack(embs)
            for q in self.queries:
                sims = m @ q

                # 1. Frame-level top-K (legacy)
                order = np.argsort(-sims)[: self.top_k]
                top_k_per_query.append([
                    {"timestamp": float(ts[i]), "score": float(sims[i])}
                    for i in order
                ])

                # 2. Segment-level aggregation via frame score sequence
                pairs = list(zip(ts, sims.tolist()))
                segs = self.aggregator.aggregate(pairs)
                segments_per_query.append([s.to_dict() for s in segs])
        else:
            top_k_per_query = [[] for _ in self.queries]
            segments_per_query = [[] for _ in self.queries]

        return TaskResult(
            task_id=self.task_id,
            payload={
                "top_k_per_query": top_k_per_query,
                "segments_per_query": segments_per_query,
            },
            metrics={
                "n_sparse_frames": float(len(self._sparse_embs)),
                "n_dense_frames": float(len(self._dense_embs)),
                "n_total_segments": float(
                    sum(len(s) for s in segments_per_query)
                ),
            },
        )
