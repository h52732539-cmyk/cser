"""Route executor — execute a single RetrievalRoute and return metrics.

Given a route + query + index + metadata, produces:
  - rank of GT video
  - recall@K
  - gt_filtered (bool: did hard filter eliminate GT?)
  - candidate_count after filtering
  - cost proxy (based on budget tier + model calls)

Calls into existing OfflineIndex / MetaFilter; does NOT reimplement search.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .route_schema import RetrievalRoute

# We import from existing core modules — no reimplementation.
import sys
from pathlib import Path
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from core.offline_index import OfflineIndex
from core.meta_filter import MetaFilter
from core.metadata import VideoMetadata
from core.query_parser import QueryIntent


# ----------------------------------------------------------------------

COST_TABLE = {"low": 1.0, "medium": 2.0, "high": 4.0, "full": 8.0}


@dataclass
class RouteResult:
    route_id: str
    rank: int                   # 0-based rank of GT (-1 if not found)
    gt_filtered: bool           # True if GT was eliminated by hard filter
    candidate_count: int        # videos remaining after hard filter
    cost_proxy: float           # budget-tier cost
    latency_ms: float = 0.0
    recall_at: Dict[int, int] = field(default_factory=dict)  # K → 0/1

    @property
    def mrr(self) -> float:
        if self.rank < 0:
            return 0.0
        return 1.0 / (self.rank + 1)

    def to_dict(self) -> Dict:
        return {
            "route_id": self.route_id,
            "rank": self.rank,
            "gt_filtered": self.gt_filtered,
            "candidate_count": self.candidate_count,
            "cost_proxy": self.cost_proxy,
            "latency_ms": self.latency_ms,
            "mrr": self.mrr,
            **{f"recall@{k}": v for k, v in self.recall_at.items()},
        }


# ----------------------------------------------------------------------

class RouteExecutor:
    """Execute a RetrievalRoute against the OfflineIndex."""

    def __init__(self,
                 index: OfflineIndex,
                 meta_filter: Optional[MetaFilter] = None,
                 alpha_nnn: float = 0.7,
                 tau_qamp: float = 0.10,
                 col_beta: float = 0.4) -> None:
        self.index = index
        self.mf = meta_filter or MetaFilter()
        self.alpha_nnn = alpha_nnn
        self.tau_qamp = tau_qamp
        self.col_beta = col_beta
        self._metas = [e.metadata for e in index.entries]
        self._id_to_idx = {e.video_id: i for i, e in enumerate(index.entries)}

    # ------------------------------------------------------------------

    def execute(self,
                route: RetrievalRoute,
                query_emb: np.ndarray,
                gt_video_id: str,
                intent: Optional[QueryIntent] = None,
                ) -> RouteResult:
        """Run one route for one query."""
        t0 = time.perf_counter()
        N = len(self.index.entries)

        # --- 1. Semantic base scores (always computed) ---
        # Use search_batch with col_beta=0 to get raw reranked scores.
        hits = self.index.search_batch(
            query_emb[np.newaxis, :],
            top_k=N,
            alpha_nnn=self.alpha_nnn,
            tau_qamp=self.tau_qamp,
            col_beta=0.0,
            topm_rerank=route.candidate_topm,
        )[0]
        sem_scores = np.zeros(N, dtype=np.float32)
        for vid, sc, _ in hits:
            sem_scores[self._id_to_idx[vid]] = sc

        # --- 2. Hard filter ---
        gt_filtered = False
        if route.has_hard_filter and intent is not None:
            filter_intent = QueryIntent(
                semantic_text="",
                time_window=intent.time_window if "time" in route.hard_axes else None,
                geo_categories=intent.geo_categories if "geo" in route.hard_axes else [],
                motion_classes=intent.motion_classes if "motion" in route.hard_axes else [],
                device_filter=intent.device_filter if "device" in route.hard_axes else None,
            )
            fr = self.mf.filter(self._metas, filter_intent)
            mask = fr.mask
            gt_idx = self._id_to_idx.get(gt_video_id)
            if gt_idx is not None and not mask[gt_idx]:
                gt_filtered = True
        else:
            mask = np.ones(N, dtype=bool)

        candidate_count = int(mask.sum())

        # --- 3. Col-softmax (post-filter if route uses it) ---
        if route.rerank_mode == "col_softmax_post_filter" and candidate_count > 1:
            masked = np.where(mask, sem_scores, -1e9)
            col_max = masked.max()
            z = (masked - col_max) / max(self.col_beta, 1e-6)
            e = np.exp(z)
            final = e / (e.sum() + 1e-12)
            final = np.where(mask, final, -1e9)
        elif route.rerank_mode == "none":
            final = np.where(mask, sem_scores, -1e9)
        else:
            final = np.where(mask, sem_scores, -1e9)

        # --- 4. Soft rerank ---
        if route.has_soft_rerank and intent is not None and candidate_count > 0:
            soft_intent = QueryIntent(
                semantic_text="",
                time_window=intent.time_window if "time" in route.soft_axes else None,
                geo_categories=intent.geo_categories if "geo" in route.soft_axes else [],
                motion_classes=intent.motion_classes if "motion" in route.soft_axes else [],
                device_filter=intent.device_filter if "device" in route.soft_axes else None,
            )
            soft = self.mf.soft_score(self._metas, soft_intent)
            masked_vals = final[mask]
            if len(masked_vals) > 0:
                s_min, s_max = masked_vals.min(), masked_vals.max()
                if s_max > s_min:
                    sem_norm = (final - s_min) / (s_max - s_min)
                else:
                    sem_norm = np.where(mask, 0.5, -1e9)
                final = np.where(mask, 0.8 * sem_norm + 0.2 * soft, -1e9)

        # --- 5. Rank GT ---
        order = np.argsort(-final)
        gt_idx = self._id_to_idx.get(gt_video_id)
        if gt_idx is None or gt_filtered:
            rank = -1
        else:
            ranked_ids = [self.index.entries[i].video_id for i in order
                          if final[i] > -1e8]
            rank = ranked_ids.index(gt_video_id) if gt_video_id in ranked_ids else -1

        dt = (time.perf_counter() - t0) * 1000.0

        recall_at = {}
        for K in (1, 5, 10):
            recall_at[K] = int(0 <= rank < K)

        return RouteResult(
            route_id=route.route_id,
            rank=rank,
            gt_filtered=gt_filtered,
            candidate_count=candidate_count,
            cost_proxy=COST_TABLE.get(route.budget_tier, 1.0),
            latency_ms=dt,
            recall_at=recall_at,
        )

    # ------------------------------------------------------------------

    def survival_labels(self,
                         gt_video_id: str,
                         intent: QueryIntent) -> Dict[str, int]:
        """For each axis, check if GT survives hard filtering."""
        gt_idx = self._id_to_idx.get(gt_video_id)
        if gt_idx is None:
            return {a: 0 for a in ("time", "geo", "motion", "device")}
        out = {}
        for axis in ("time", "geo", "motion", "device"):
            ax_intent = QueryIntent(
                semantic_text="",
                time_window=intent.time_window if axis == "time" else None,
                geo_categories=intent.geo_categories if axis == "geo" else [],
                motion_classes=intent.motion_classes if axis == "motion" else [],
                device_filter=intent.device_filter if axis == "device" else None,
            )
            if not ax_intent.has_constraint():
                out[axis] = 1
                continue
            fr = self.mf.filter(self._metas, ax_intent)
            out[axis] = int(bool(fr.mask[gt_idx]))
        return out
