"""Two-stage controller: aggregates Stage-1 interest signals into Stage-2 intervals."""
from __future__ import annotations

from typing import List

from .types import InterestSignal, Interval


class TwoStageController:
    """Merge interest intervals from all Stage-1 tasks into a ranked list."""

    def __init__(
        self,
        max_intervals: int = 10,
        interval_expand_sec: float = 0.5,
        merge_gap_sec: float = 1.0,
        max_total_duration_sec: float = 60.0,
    ) -> None:
        self.max_intervals = max_intervals
        self.expand = interval_expand_sec
        self.merge_gap = merge_gap_sec
        self.max_total = max_total_duration_sec

    def aggregate(self, signals: List[InterestSignal]) -> List[Interval]:
        raw: List[Interval] = []
        for sig in signals:
            for iv in sig.intervals:
                raw.append(Interval(
                    start=max(0.0, iv.start - self.expand),
                    end=iv.end + self.expand,
                    score=float(iv.score),
                    source_task=sig.task_id,
                ))
        if not raw:
            return []

        # Sort by start time, merge overlapping / close intervals.
        raw.sort(key=lambda x: x.start)
        merged: List[Interval] = [raw[0]]
        for iv in raw[1:]:
            last = merged[-1]
            if iv.start - last.end <= self.merge_gap:
                merged[-1] = Interval(
                    start=last.start,
                    end=max(last.end, iv.end),
                    score=max(last.score, iv.score),
                    source_task=last.source_task + "+" + iv.source_task,
                )
            else:
                merged.append(iv)

        # Rank by score, enforce caps.
        merged.sort(key=lambda x: -x.score)
        picked: List[Interval] = []
        total = 0.0
        for iv in merged:
            if len(picked) >= self.max_intervals:
                break
            if total + iv.length() > self.max_total:
                # shrink to remaining budget or skip
                remaining = self.max_total - total
                if remaining < 0.2:
                    continue
                iv = Interval(
                    start=iv.start,
                    end=iv.start + remaining,
                    score=iv.score,
                    source_task=iv.source_task,
                )
            picked.append(iv)
            total += iv.length()

        # Return in temporal order for nicer downstream logs.
        picked.sort(key=lambda x: x.start)
        return picked
