"""SegmentAggregator: convert frame-level scores to [t_start, t_end] segments.

Shared by all tasks that need to produce temporal segments as final output.
Pipeline:
    raw (timestamp, score) pairs
        -> sort by time
        -> smooth (moving average)
        -> threshold
        -> group above-threshold consecutive points into segments
        -> merge segments closer than `merge_gap_sec`
        -> temporal NMS
        -> keep top-K segments by score
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Segment:
    """A temporal segment [start, end] with an aggregated score."""
    start: float
    end: float
    score: float

    def length(self) -> float:
        return max(0.0, self.end - self.start)

    def iou(self, other: "Segment") -> float:
        s = max(self.start, other.start)
        e = min(self.end, other.end)
        inter = max(0.0, e - s)
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        return {"start": float(self.start),
                "end": float(self.end),
                "score": float(self.score)}


class SegmentAggregator:
    """Frame-level scores -> [t_start, t_end] segments.

    Args:
        percentile: threshold is the p-th percentile of the input scores
            (used only if absolute_threshold is None). E.g. 0.85 keeps top 15%.
        absolute_threshold: if not None, overrides percentile threshold.
        smooth_window: moving-average window (in frames). 1 disables smoothing.
        merge_gap_sec: merge segments whose end/start are closer than this.
        min_segment_sec: drop segments shorter than this.
        max_segments: keep at most this many segments (by score).
        nms_iou: temporal NMS threshold. 0 disables NMS.
        pad_sec: pad each segment boundary by this (seconds).
    """

    def __init__(
        self,
        percentile: float = 0.80,
        absolute_threshold: float | None = None,
        smooth_window: int = 3,
        merge_gap_sec: float = 0.8,
        min_segment_sec: float = 0.3,
        max_segments: int = 5,
        nms_iou: float = 0.5,
        pad_sec: float = 0.0,
    ) -> None:
        self.percentile = percentile
        self.abs_thr = absolute_threshold
        self.smooth_w = max(1, int(smooth_window))
        self.merge_gap = float(merge_gap_sec)
        self.min_len = float(min_segment_sec)
        self.max_segs = int(max_segments)
        self.nms_iou = float(nms_iou)
        self.pad_sec = float(pad_sec)

    # ------------------------------------------------------------------

    def aggregate(
        self,
        pairs: Sequence[Tuple[float, float]],
    ) -> List[Segment]:
        """Main entry. `pairs` is list of (timestamp_sec, score)."""
        if not pairs:
            return []
        pairs = sorted(pairs, key=lambda x: x[0])
        ts = np.array([p[0] for p in pairs], dtype=np.float64)
        sc = np.array([p[1] for p in pairs], dtype=np.float64)

        sc = self._smooth(sc, self.smooth_w)
        thr = self._threshold(sc)

        # Group consecutive points above threshold into raw segments.
        raw: List[Segment] = []
        in_seg = False
        seg_start = 0.0
        seg_peak = -np.inf
        prev_t = ts[0]
        for i, (t, s) in enumerate(zip(ts, sc)):
            if s >= thr:
                if not in_seg:
                    seg_start = float(t)
                    seg_peak = float(s)
                    in_seg = True
                else:
                    seg_peak = max(seg_peak, float(s))
                prev_t = float(t)
            else:
                if in_seg:
                    raw.append(Segment(
                        start=seg_start,
                        end=float(prev_t),
                        score=seg_peak,
                    ))
                    in_seg = False
        if in_seg:
            raw.append(Segment(
                start=seg_start,
                end=float(ts[-1]),
                score=seg_peak,
            ))

        raw = self._merge(raw, self.merge_gap)
        raw = [seg for seg in raw if seg.length() >= self.min_len]

        if self.pad_sec > 0:
            raw = [
                Segment(
                    start=max(0.0, seg.start - self.pad_sec),
                    end=seg.end + self.pad_sec,
                    score=seg.score,
                )
                for seg in raw
            ]

        if self.nms_iou > 0 and len(raw) > 1:
            raw = self._nms(raw, self.nms_iou)

        raw.sort(key=lambda s: -s.score)
        return raw[: self.max_segs]

    # ------------------------------------------------------------------

    @staticmethod
    def _smooth(scores: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return scores
        kernel = np.ones(w, dtype=np.float64) / float(w)
        return np.convolve(scores, kernel, mode="same")

    def _threshold(self, scores: np.ndarray) -> float:
        if self.abs_thr is not None:
            return float(self.abs_thr)
        if len(scores) == 0:
            return 0.0
        return float(np.quantile(scores, self.percentile))

    @staticmethod
    def _merge(segs: List[Segment], gap: float) -> List[Segment]:
        if not segs:
            return []
        segs = sorted(segs, key=lambda s: s.start)
        out = [segs[0]]
        for seg in segs[1:]:
            last = out[-1]
            if seg.start - last.end <= gap:
                out[-1] = Segment(
                    start=last.start,
                    end=max(last.end, seg.end),
                    score=max(last.score, seg.score),
                )
            else:
                out.append(seg)
        return out

    @staticmethod
    def _nms(segs: List[Segment], iou_thr: float) -> List[Segment]:
        segs = sorted(segs, key=lambda s: -s.score)
        kept: List[Segment] = []
        for s in segs:
            if all(s.iou(k) < iou_thr for k in kept):
                kept.append(s)
        return kept


# ----------------------------------------------------------------------
#  Utilities
# ----------------------------------------------------------------------

def segments_mean_iou(
    pred: Sequence[Segment],
    gt: Sequence[Segment],
) -> float:
    """Bidirectional segment-level mean IoU (F1-style)."""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    p_best = [max(p.iou(g) for g in gt) for p in pred]
    g_best = [max(g.iou(p) for p in pred) for g in gt]
    return float((np.mean(p_best) + np.mean(g_best)) / 2.0)


def boundary_mae(
    pred: Sequence[Segment],
    gt: Sequence[Segment],
) -> float:
    """Mean absolute error in seconds between matched segment boundaries.

    For each gt segment, match its closest pred by IoU; report average of
    |start delta| + |end delta|.
    """
    if not pred or not gt:
        return float("inf") if (pred or gt) else 0.0
    errs: List[float] = []
    for g in gt:
        best = max(pred, key=lambda p: p.iou(g))
        errs.append(abs(best.start - g.start) + abs(best.end - g.end))
    return float(np.mean(errs))
