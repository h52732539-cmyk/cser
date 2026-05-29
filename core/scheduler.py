"""Unified multi-task sampling scheduler.

Core responsibility: given N task subscriptions, fuse their desired sampling
plans into a single deduplicated FrameRequest list that can be consumed by a
shared decode loop.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from .prefilter import PrefilterResult
from .subscription import TaskSubscription
from .types import FrameRequest, Interval, SamplingStage


class UnifiedScheduler:
    """Multi-task budget fusion + two-stage plan generator."""

    def __init__(
        self,
        subscriptions: List[TaskSubscription],
        merge_gap_sec: float = 0.05,
        assumed_video_fps: float = 25.0,
    ) -> None:
        self.subs: Dict[str, TaskSubscription] = {
            s.task_id: s for s in subscriptions
        }
        self.merge_gap = merge_gap_sec
        self.assumed_fps = assumed_video_fps

    # ---- Stage 1 plan -------------------------------------------------

    def plan_sparse(
        self,
        video_id: str,
        duration: float,
        prefilter: Optional[PrefilterResult] = None,
    ) -> List[FrameRequest]:
        """Generate the Stage-1 (sparse preview) sampling plan."""
        proposals: Dict[str, List[float]] = {}
        for task_id, sub in self.subs.items():
            # Gated tasks do not run in Stage 1
            if sub.gated_by is not None:
                continue
            if sub.sparse_fps <= 0.0:
                continue
            stride = 1.0 / sub.sparse_fps
            times = list(np.arange(0.0, max(duration, 1e-6), stride))
            if len(times) > sub.max_frames_sparse:
                times = list(np.linspace(
                    0.0, max(duration, 1e-6), sub.max_frames_sparse,
                    endpoint=False
                ))
            proposals[task_id] = [float(t) for t in times]
        return self._merge_proposals(
            video_id, proposals, prefilter, SamplingStage.SPARSE
        )

    # ---- Stage 2 plan -------------------------------------------------

    def plan_dense(
        self,
        video_id: str,
        intervals: List[Interval],
        prefilter: Optional[PrefilterResult] = None,
    ) -> List[FrameRequest]:
        """Generate the Stage-2 (dense refine) sampling plan."""
        if not intervals:
            return []
        proposals: Dict[str, List[float]] = {}
        for task_id, sub in self.subs.items():
            # Only full-path tasks (can_produce_interest or gated) run Stage 2
            if not sub.can_produce_interest and sub.gated_by is None:
                continue
            if sub.dense_fps <= 0.0:
                continue
            stride = 1.0 / sub.dense_fps
            task_times: List[float] = []
            for iv in intervals:
                ts = list(np.arange(iv.start, iv.end, stride))
                task_times.extend(ts)
            task_times = sorted(set(round(float(t), 3) for t in task_times))
            if len(task_times) > sub.max_frames_dense:
                task_times = task_times[:sub.max_frames_dense]
            proposals[task_id] = task_times
        return self._merge_proposals(
            video_id, proposals, prefilter, SamplingStage.DENSE
        )

    # ---- Fusion core --------------------------------------------------

    def _merge_proposals(
        self,
        video_id: str,
        proposals: Dict[str, List[float]],
        prefilter: Optional[PrefilterResult],
        stage: SamplingStage,
    ) -> List[FrameRequest]:
        if not proposals:
            return []

        all_points: List[tuple] = []
        for task_id, times in proposals.items():
            for t in times:
                all_points.append((float(t), task_id))
        if not all_points:
            return []
        all_points.sort(key=lambda x: x[0])

        # 1. merge nearby timestamps that fall within merge_gap
        merged: List[tuple] = []
        cur_t, first_task = all_points[0]
        cur_tasks: Set[str] = {first_task}
        for t, tid in all_points[1:]:
            if t - cur_t < self.merge_gap:
                cur_tasks.add(tid)
            else:
                merged.append((cur_t, cur_tasks))
                cur_t = t
                cur_tasks = {tid}
        merged.append((cur_t, cur_tasks))

        # 2. apply prefilter mask
        if prefilter is not None and prefilter.candidate_mask is not None:
            mask = prefilter.candidate_mask
            T = int(len(mask))
            filtered: List[tuple] = []
            for t, tasks in merged:
                # A task may opt out of metadata prefiltering
                if any(not self.subs[tid].respects_metadata for tid in tasks):
                    filtered.append((t, tasks))
                    continue
                idx = int(t * 10)
                if 0 <= idx < T and mask[idx]:
                    filtered.append((t, tasks))
            merged = filtered

        # 3. deterministic ordering
        merged.sort(key=lambda x: x[0])

        requests: List[FrameRequest] = []
        for t, tasks in merged:
            requests.append(FrameRequest(
                video_id=video_id,
                frame_idx=int(t * self.assumed_fps),
                timestamp=float(t),
                stage=stage,
                subscribers=set(tasks),
            ))
        return requests
