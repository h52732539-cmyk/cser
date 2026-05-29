"""MetaFilter — numpy-only metadata-based candidate filtering and
post-hoc score fusion.

Two modes:

  1. `filter(intent)` — hard filter: returns a bool mask over the
     OfflineIndex entries. Videos failing any specified constraint are
     excluded. Used to shrink the candidate set 5-50× before the dense
     semantic search.

  2. `soft_score(intent)` — soft score: returns a [N] array in [0, 1]
     reflecting how well each video matches the metadata constraint.
     Used in `α·semantic + β·meta` post-hoc fusion.

Both run in pure numpy on the existing index; no model call involved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .metadata import VideoMetadata
from .query_parser import QueryIntent


# ----------------------------------------------------------------------
#  Haversine distance (for GPS proximity — not used yet but exposed)
# ----------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------
#  MetaFilter
# ----------------------------------------------------------------------

@dataclass
class FilterResult:
    mask: np.ndarray            # bool (N,) True = keep
    n_kept: int
    n_total: int
    fired_constraints: List[str]


class MetaFilter:
    """Filter / score OfflineIndex entries by metadata.

    Safe when metadata is partially missing: a video with an absent
    field is kept if the intent applies to that field, unless the
    strict-mode flag is set.
    """

    def __init__(self,
                 time_slack_sec: float = 3600.0,
                 strict: bool = False) -> None:
        self.time_slack = float(time_slack_sec)
        self.strict = bool(strict)

    # ------------------------------------------------------------------

    def filter(self,
               metas: Sequence[Optional[VideoMetadata]],
               intent: QueryIntent) -> FilterResult:
        N = len(metas)
        mask = np.ones(N, dtype=bool)
        fired: List[str] = []
        if not intent.has_constraint():
            return FilterResult(mask=mask, n_kept=N, n_total=N,
                                fired_constraints=fired)

        # --- time window ---
        if intent.time_window is not None:
            s, e = intent.time_window
            s -= self.time_slack
            e += self.time_slack
            fired.append("time")
            for i, m in enumerate(metas):
                if m is None or m.creation_time is None:
                    mask[i] &= not self.strict
                else:
                    mask[i] &= (s <= m.creation_time <= e)

        # --- geo category ---
        if intent.geo_categories:
            want = set(intent.geo_categories)
            fired.append("geo")
            for i, m in enumerate(metas):
                if m is None or m.geo_category is None:
                    mask[i] &= not self.strict
                else:
                    mask[i] &= (m.geo_category in want)

        # --- motion class ---
        if intent.motion_classes:
            want = set(intent.motion_classes)
            fired.append("motion")
            for i, m in enumerate(metas):
                if m is None or m.motion_class is None:
                    mask[i] &= not self.strict
                else:
                    mask[i] &= (m.motion_class in want)

        # --- device make ---
        if intent.device_filter:
            want = intent.device_filter.lower()
            fired.append("device")
            for i, m in enumerate(metas):
                if m is None or not m.device_make:
                    mask[i] &= not self.strict
                else:
                    mask[i] &= (want in m.device_make.lower())

        return FilterResult(mask=mask, n_kept=int(mask.sum()),
                            n_total=N, fired_constraints=fired)

    # ------------------------------------------------------------------

    def soft_score(self,
                    metas: Sequence[Optional[VideoMetadata]],
                    intent: QueryIntent) -> np.ndarray:
        """Return a (N,) score in [0, 1] reflecting metadata match."""
        N = len(metas)
        if not intent.has_constraint():
            return np.ones(N, dtype=np.float32)

        scores = np.ones(N, dtype=np.float32)
        n_axes = 0

        # Time: trapezoidal window with slack soft edges
        if intent.time_window is not None:
            s, e = intent.time_window
            center = (s + e) / 2
            half = max((e - s) / 2, 1.0)
            slack = max(self.time_slack, half)
            t_score = np.zeros(N, dtype=np.float32)
            for i, m in enumerate(metas):
                if m is None or m.creation_time is None:
                    t_score[i] = 0.5        # neutral
                    continue
                d = abs(m.creation_time - center)
                if d <= half:
                    t_score[i] = 1.0
                elif d >= half + slack:
                    t_score[i] = 0.0
                else:
                    t_score[i] = 1.0 - (d - half) / slack
            scores *= t_score
            n_axes += 1

        if intent.geo_categories:
            want = set(intent.geo_categories)
            g_score = np.zeros(N, dtype=np.float32)
            for i, m in enumerate(metas):
                if m is None or m.geo_category is None:
                    g_score[i] = 0.5
                elif m.geo_category in want:
                    g_score[i] = 1.0
            scores *= g_score
            n_axes += 1

        if intent.motion_classes:
            want = set(intent.motion_classes)
            mo_score = np.zeros(N, dtype=np.float32)
            for i, m in enumerate(metas):
                if m is None or m.motion_class is None:
                    mo_score[i] = 0.5
                elif m.motion_class in want:
                    # weight by motion_confidence if available
                    mo_score[i] = float(m.motion_confidence or 1.0)
            scores *= mo_score
            n_axes += 1

        if intent.device_filter:
            want = intent.device_filter.lower()
            d_score = np.zeros(N, dtype=np.float32)
            for i, m in enumerate(metas):
                if m is None or not m.device_make:
                    d_score[i] = 0.5
                elif want in m.device_make.lower():
                    d_score[i] = 1.0
            scores *= d_score
            n_axes += 1

        # Rescale geometric-mean-like so multiple axes don't overwhelm
        if n_axes > 1:
            scores = scores ** (1.0 / n_axes)
        return scores


# ----------------------------------------------------------------------
#  Hybrid semantic + metadata fusion
# ----------------------------------------------------------------------

def fuse_scores(semantic: np.ndarray,
                meta_soft: np.ndarray,
                alpha: float = 0.7) -> np.ndarray:
    """Convex blend: α·semantic + (1-α)·meta.

    `semantic` and `meta_soft` are assumed both in [0, 1] and of
    identical length N.
    """
    sem = np.asarray(semantic, dtype=np.float32)
    met = np.asarray(meta_soft, dtype=np.float32)
    # Normalize semantic into [0,1] if already softmax-ed just clamp
    s_min, s_max = float(sem.min()), float(sem.max())
    if s_max > s_min:
        sem = (sem - s_min) / (s_max - s_min)
    return alpha * sem + (1.0 - alpha) * met
