"""Retrieval value function f(S, q) over real expert signals.

The Submodular Value Network learns to predict marginal values of this function.
Given a query and a *selection* of optional experts, it returns a scalar value
in [0, 1] measuring retrieval quality (reciprocal rank of the GT video).

Scoring model
-------------
Every video carries cached expert signals (``GallerySignals``, produced once by
``expert_features.extract_gallery_signals``). For a query with priors
``QueryExpertPriors``:

* **semantic base (always on):** ``cos(text_emb, video.clip_mean)``, min-max
  normalised to [0, 1] over the gallery.
* **highlight:** add the per-video highlight prior, weighted up if the query asks
  for action/highlights (else a mild query-agnostic prior).
* **face:** if the query implies a person, boost videos that contain a face
  (SCRFD confidence); otherwise contributes ~nothing.
* **face_id:** rerank by ArcFace-embedding similarity to the query's face prior
  (only meaningful when a reference face is given; otherwise neutral).
* **scene:** boost videos whose dominant scene matches the query's scene cue.

Selected experts are blended additively into the score; the value lattice never
removes the GT video, so f(S,q) is well defined for all 16 subsets. The integrated
pipeline may apply a candidate mask after scoring; protecting that mask is the
Conformal Safety Gate's job.

This design gives experts genuinely *overlapping, complementary* value
(face + scene both fire on "a person at the beach"), which is the source of the
submodularity the paper studies.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .expert_features import GallerySignals, QueryExpertPriors
from .experts import OPTIONAL_NAMES


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


class RetrievalEngine:
    """Compute f(S, q) for arbitrary expert selections over one gallery.

    All expensive model work is done once up-front in ``GallerySignals``; this
    class only does cheap vector arithmetic per query, so enumerating all 16
    subsets per query is fast.
    """

    def __init__(self,
                 gallery: GallerySignals,
                 expert_weight: float = 0.35) -> None:
        self.g = gallery
        self.w = float(expert_weight)
        self._N = gallery.size
        self._id_to_idx = {v: i for i, v in enumerate(gallery.video_ids)}
        # Precompute per-expert gallery score vectors that are query-independent.
        self._clip = gallery.clip_matrix()                  # (N, D)
        self._highlight = gallery.highlight_vector()         # (N,)
        self._face = gallery.face_vector()                   # (N,)
        self._face_emb = gallery.face_emb_matrix()           # (N, Df)
        self._scene_labels = [gallery.scene_label_of(i) for i in range(self._N)]

    # ------------------------------------------------------------------
    #  Per-expert query-conditioned score vectors (each length N, in [0,1])
    # ------------------------------------------------------------------

    def semantic_scores(self, priors: QueryExpertPriors) -> np.ndarray:
        sims = self._clip @ priors.text_emb                  # cosine (unit vecs)
        return _minmax(sims).astype(np.float32)

    def _expert_score(self, name: str, priors: QueryExpertPriors) -> np.ndarray:
        if name == "highlight":
            base = _minmax(self._highlight)
            gain = 1.0 if priors.wants_highlight else 0.3
            return (gain * base).astype(np.float32)
        if name == "face":
            if not priors.wants_person:
                return np.zeros(self._N, np.float32)
            return _minmax(self._face).astype(np.float32)
        if name == "face_id":
            if priors.face_emb is None:
                return np.zeros(self._N, np.float32)
            ref = priors.face_emb / (np.linalg.norm(priors.face_emb) + 1e-9)
            sims = self._face_emb @ ref
            return _minmax(sims).astype(np.float32)
        if name == "scene":
            if priors.scene_cue is None:
                return np.zeros(self._N, np.float32)
            match = np.array([1.0 if lb == priors.scene_cue else 0.0
                              for lb in self._scene_labels], dtype=np.float32)
            return match
        raise ValueError(f"unknown expert '{name}'")

    # ------------------------------------------------------------------

    def final_scores(self, priors: QueryExpertPriors,
                     active_experts: Sequence[str]) -> np.ndarray:
        """Full (N,) score vector for a selection (semantic base + experts)."""
        score = self.semantic_scores(priors).copy()
        for name in active_experts:
            score = score + self.w * self._expert_score(name, priors)
        return score

    def ranked_ids(self, scores: np.ndarray) -> List[str]:
        order = np.argsort(-scores)
        return [self.g.video_ids[i] for i in order]

    def semantic_norm(self, priors: QueryExpertPriors) -> np.ndarray:
        """Normalised semantic similarity in [0,1] — basis of the conformal score."""
        return self.semantic_scores(priors)

    def semantic_top_k_mask(self, sim_norm: np.ndarray, top_k: int) -> np.ndarray:
        """Keep the strongest semantic candidates before expert reranking."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = np.asarray(sim_norm)
        if scores.shape != (self._N,):
            raise ValueError(f"sim_norm must have shape ({self._N},)")
        if top_k >= self._N:
            return np.ones(self._N, dtype=bool)
        keep = np.argpartition(scores, -top_k)[-top_k:]
        mask = np.zeros(self._N, dtype=bool)
        mask[keep] = True
        return mask

    def id_to_idx(self, video_id: str) -> int:
        return self._id_to_idx.get(video_id, -1)

    def rank_of_gt(self, priors: QueryExpertPriors, gt_video_id: str,
                   active_experts: Sequence[str],
                   candidate_mask: Optional[np.ndarray] = None) -> int:
        gi = self._id_to_idx.get(gt_video_id, -1)
        if gi < 0:
            return -1
        final = self.final_scores(priors, active_experts)
        if candidate_mask is None:
            mask = np.ones(self._N, dtype=bool)
        else:
            mask = np.asarray(candidate_mask, dtype=bool)
            if mask.shape != (self._N,):
                raise ValueError(f"candidate_mask must have shape ({self._N},)")
        if not mask[gi]:
            return -1
        return int((final[mask] > final[gi]).sum())

    def value(self, priors: QueryExpertPriors, gt_video_id: str,
              active_experts: Sequence[str], metric: str = "rr") -> float:
        rank = self.rank_of_gt(priors, gt_video_id, active_experts)
        if rank < 0:
            return 0.0
        if metric == "rr":
            return 1.0 / (rank + 1.0)
        if metric == "recall@1":
            return float(rank < 1)
        if metric == "recall@5":
            return float(rank < 5)
        if metric == "recall@10":
            return float(rank < 10)
        raise ValueError(f"unknown metric '{metric}'")
