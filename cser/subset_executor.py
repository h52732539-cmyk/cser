"""Execute selected CSER expert subsets."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .expert_store import ExpertOutputStore
from .schema import CSERSubsetResult, DEFAULT_EXPERTS, ExpertSpec, expert_cost


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-8:
        out[finite] = 0.0
    else:
        out[finite] = (vals - lo) / (hi - lo)
    return out


class CSERSubsetExecutor:
    """Fuses selected expert outputs and ranks the gallery."""

    def __init__(
        self,
        store: ExpertOutputStore,
        expert_specs: Sequence[ExpertSpec] = DEFAULT_EXPERTS,
        fusion_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.store = store
        self.expert_specs = tuple(expert_specs)
        self.expert_ids = tuple(spec.expert_id for spec in self.expert_specs)
        self.fusion_weights = dict(fusion_weights or {})

    def subset_cost(self, selected_experts: Sequence[str]) -> float:
        return float(sum(expert_cost(e, self.expert_specs) for e in set(selected_experts)))

    def semantic_scores(self, query_emb: np.ndarray) -> np.ndarray:
        return self.store.score_expert("clip_semantic", query_emb).scores

    def execute_subset(
        self,
        selected_experts: Sequence[str],
        query_emb: np.ndarray,
        gt_video_id: str,
        query_context: Optional[Mapping[str, object]] = None,
        protected_mask: Optional[np.ndarray] = None,
        recall_ks: Sequence[int] = (1, 5, 10),
    ) -> CSERSubsetResult:
        selected = tuple(dict.fromkeys(selected_experts))
        n = self.store.size
        if protected_mask is None:
            protected = np.zeros(n, dtype=bool)
        else:
            protected = np.asarray(protected_mask, dtype=bool).reshape(n)

        fused = np.zeros(n, dtype=np.float32)
        keep = np.ones(n, dtype=bool)
        total_weight = 0.0

        for expert_id in selected:
            expert_score = self.store.score_expert(expert_id, query_emb, query_context)
            weight = float(self.fusion_weights.get(expert_id, 1.0))
            fused += weight * normalize_scores(expert_score.scores)
            total_weight += weight
            if expert_score.keep_mask is not None:
                keep &= (expert_score.keep_mask | protected)

        if total_weight > 0:
            fused /= total_weight

        candidate_count = int(keep.sum())
        gt_idx = self.store.index_of(gt_video_id) if gt_video_id in self.store._id_to_idx else -1
        gt_filtered = bool(gt_idx < 0 or not keep[gt_idx])

        final_scores = np.where(keep, fused, -1e9).astype(np.float32)
        rank = -1
        if gt_idx >= 0 and not gt_filtered and candidate_count > 0:
            order = np.argsort(-final_scores)
            ranked = [int(i) for i in order if final_scores[int(i)] > -1e8]
            rank = ranked.index(gt_idx) if gt_idx in ranked else -1

        recall_at: Dict[int, int] = {
            int(k): int(rank >= 0 and rank < int(k)) for k in recall_ks
        }
        mrr = 0.0 if rank < 0 else 1.0 / float(rank + 1)

        return CSERSubsetResult(
            selected_experts=selected,
            rank=int(rank),
            recall_at=recall_at,
            mrr=float(mrr),
            gt_filtered=gt_filtered,
            cost=self.subset_cost(selected),
            candidate_count=candidate_count,
            scores=final_scores,
            keep_mask=keep,
        )
