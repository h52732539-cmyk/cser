"""Main framework orchestrator — runs the two-stage multi-task pipeline."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .cache import SharedFrameCache
from .decoder import decode_frames
from .prefilter import MetadataPrefilter, PrefilterResult
from .scheduler import UnifiedScheduler
from .subscription import TaskSubscription
from .two_stage import TwoStageController
from .types import Frame, InterestSignal, SamplingStage, TaskResult


class LiteVTRFramework:
    """Multi-model unified sampling framework.

    Lifecycle (per video):
        Stage 0  Metadata prefilter    -> PrefilterResult
        Stage 1  Sparse preview        -> emit InterestSignal[]
        Stage 2  Dense refine          -> full task execution
                 (skipped if no interest signals or if disabled)
        Stage 3  Finalize + gating     -> TaskResult per task
    """

    def __init__(
        self,
        tasks,
        prefilter: Optional[MetadataPrefilter] = None,
        two_stage_controller: Optional[TwoStageController] = None,
        cache_size: int = 500,
        enable_two_stage: bool = True,
        enable_prefilter: bool = True,
    ) -> None:
        self.tasks = {t.task_id: t for t in tasks}
        self.scheduler = UnifiedScheduler([t.sub for t in tasks])
        self.prefilter = prefilter or MetadataPrefilter()
        self.two_stage = two_stage_controller or TwoStageController()
        self.cache = SharedFrameCache(cache_size)
        self.enable_two_stage = enable_two_stage
        self.enable_prefilter = enable_prefilter
        self.stats: Dict[str, float] = {}

    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str,
        duration: float,
        video_id: str,
        sensor_stream: Optional[dict] = None,
    ) -> Dict[str, TaskResult]:
        self._reset_stats()
        self.cache.clear()
        for t in self.tasks.values():
            t.reset()

        # ---- Stage 0: prefilter --------------------------------------
        t0 = time.perf_counter()
        if self.enable_prefilter:
            prefilter_res = self.prefilter.analyze(
                video_path, duration, sensor_stream
            )
        else:
            prefilter_res = self._identity_prefilter(duration)
        self.stats["prefilter_ms"] = (time.perf_counter() - t0) * 1000.0
        self.stats["n_static_segments"] = len(prefilter_res.static_segments)
        self.stats["n_scene_boundaries"] = len(prefilter_res.scene_boundaries)
        self.stats["n_candidate_buckets"] = prefilter_res.num_candidates()

        # ---- Stage 1: sparse preview ---------------------------------
        t0 = time.perf_counter()
        sparse_reqs = self.scheduler.plan_sparse(
            video_id, duration, prefilter_res
        )
        sparse_frames = decode_frames(video_path, sparse_reqs, self.cache)
        self.stats["stage1_decode_ms"] = (time.perf_counter() - t0) * 1000.0
        self.stats["n_stage1_frames"] = len(sparse_frames)

        t0 = time.perf_counter()
        signals = self._dispatch_sparse(sparse_frames, sparse_reqs)
        self.stats["stage1_compute_ms"] = (time.perf_counter() - t0) * 1000.0
        self.stats["n_interest_signals"] = sum(
            len(s.intervals) for s in signals
        )

        # ---- Stage 2: dense refine -----------------------------------
        dense_frames: List[Frame] = []
        if self.enable_two_stage and signals:
            intervals = self.two_stage.aggregate(signals)
            self.stats["n_intervals"] = len(intervals)
            self.stats["total_interval_sec"] = sum(
                iv.length() for iv in intervals
            )

            t0 = time.perf_counter()
            dense_reqs = self.scheduler.plan_dense(
                video_id, intervals, prefilter_res
            )
            dense_frames = decode_frames(video_path, dense_reqs, self.cache)
            self.stats["stage2_decode_ms"] = (time.perf_counter() - t0) * 1000.0
            self.stats["n_stage2_frames"] = len(dense_frames)

            t0 = time.perf_counter()
            self._dispatch_dense(dense_frames, dense_reqs)
            self.stats["stage2_compute_ms"] = (time.perf_counter() - t0) * 1000.0
        else:
            self.stats["n_intervals"] = 0
            self.stats["total_interval_sec"] = 0.0
            self.stats["stage2_decode_ms"] = 0.0
            self.stats["n_stage2_frames"] = 0
            self.stats["stage2_compute_ms"] = 0.0

        # ---- Stage 3: finalize ---------------------------------------
        results: Dict[str, TaskResult] = {}
        for tid, task in self.tasks.items():
            results[tid] = task.finalize()

        self.stats["total_decoded_frames"] = (
            self.stats["n_stage1_frames"] + self.stats["n_stage2_frames"]
        )
        self.stats["cache_hits"] = self.cache.hits
        self.stats["cache_misses"] = self.cache.misses
        self.stats["total_ms"] = (
            self.stats["prefilter_ms"]
            + self.stats["stage1_decode_ms"]
            + self.stats["stage1_compute_ms"]
            + self.stats["stage2_decode_ms"]
            + self.stats["stage2_compute_ms"]
        )
        return results

    # ------------------------------------------------------------------

    def _dispatch_sparse(
        self, frames: List[Frame], requests
    ) -> List[InterestSignal]:
        if not frames:
            return []
        # Build index: req -> frame (aligned lists)
        req_by_idx = {(r.frame_idx, r.timestamp): r for r in requests}

        signals: List[InterestSignal] = []
        for tid, task in self.tasks.items():
            if task.sub.gated_by is not None:
                continue
            subscribed = [
                f for f in frames
                if tid in req_by_idx.get((f.frame_idx, f.timestamp),
                                          requests[0]).subscribers
            ]
            if not subscribed:
                continue
            sig = task.process_sparse(subscribed)
            if sig is not None and task.sub.can_produce_interest:
                signals.append(sig)
        return signals

    def _dispatch_dense(self, frames: List[Frame], requests) -> None:
        if not frames:
            return
        req_by_idx = {(r.frame_idx, r.timestamp): r for r in requests}
        for tid, task in self.tasks.items():
            subscribed = [
                f for f in frames
                if tid in req_by_idx.get((f.frame_idx, f.timestamp),
                                          requests[0]).subscribers
            ]
            if subscribed:
                task.process_dense(subscribed)

    def _identity_prefilter(self, duration: float) -> PrefilterResult:
        T = max(1, int(duration * 10))
        return PrefilterResult(
            candidate_mask=np.ones(T, dtype=bool),
            scene_boundaries=[],
            static_segments=[],
        )

    def _reset_stats(self) -> None:
        self.stats = {
            "prefilter_ms": 0.0,
            "stage1_decode_ms": 0.0,
            "stage1_compute_ms": 0.0,
            "stage2_decode_ms": 0.0,
            "stage2_compute_ms": 0.0,
            "n_stage1_frames": 0,
            "n_stage2_frames": 0,
            "total_decoded_frames": 0,
            "n_static_segments": 0,
            "n_scene_boundaries": 0,
            "n_candidate_buckets": 0,
            "n_interest_signals": 0,
            "n_intervals": 0,
            "total_interval_sec": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_ms": 0.0,
        }
