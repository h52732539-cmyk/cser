"""Oracle marginal-value labeller (plan §4.2 training signal, §10 Week1-2).

For each query we enumerate **all** 2^K0 selections over the optional experts
(K0 = 4 -> 16 subsets) and record the exact value f(S, q). The lattice is small
so this is exact — no Monte-Carlo subset sampling for the labels.

From the value matrix we derive, for every (query, set S, candidate expert e):

    marginal value   v*(e | S, q) = f(S ∪ {e}, q) - f(S, q)

the regression target for the Submodular Value Network. We also emit a compact
per-query feature (CLIP text embedding + QPP stats over the semantic scores +
query-prior cue indicators) used as the SVN query input.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .expert_features import QueryExpertPriors
from .experts import (N_OPTIONAL, all_optional_masks, mask_to_names,
                      OPTIONAL_NAMES)
from .retrieval import RetrievalEngine


# Query feature = CLIP text emb (D) + QPP stats (6) + cue indicators (4)
CLIP_FEATURE_DIM = 512


def query_feature_dim(clip_dim: int) -> int:
    return clip_dim + 6 + 4


def extract_query_feature(priors: QueryExpertPriors,
                          sem_scores: np.ndarray) -> np.ndarray:
    """Build the SVN query feature from priors + the semantic score vector."""
    clip = np.asarray(priors.text_emb, dtype=np.float32).ravel()

    s = np.sort(sem_scores)[::-1]
    top1 = float(s[0]) if len(s) else 0.0
    top2 = float(s[1]) if len(s) > 1 else 0.0
    margin = top1 - top2
    mean = float(s.mean()) if len(s) else 0.0
    std = float(s.std()) if len(s) else 0.0
    # entropy of the top-20 softmax (query-difficulty proxy)
    top = s[:20]
    if len(top):
        z = top - top.max()
        p = np.exp(z); p /= p.sum() + 1e-12
        ent = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        ent = 0.0
    qpp = np.array([top1, top2, margin, mean, std, ent], dtype=np.float32)

    cues = np.array([
        float(priors.wants_person),
        float(priors.wants_highlight),
        float(priors.scene_cue is not None),
        float(priors.face_emb is not None),
    ], dtype=np.float32)

    return np.concatenate([clip, qpp, cues]).astype(np.float32)


@dataclass
class OracleLabels:
    metric: str
    query_feats: np.ndarray         # (Nq, query_feature_dim)
    value_matrix: np.ndarray        # (Nq, 2**K0)
    marginal: np.ndarray            # (Nq, 2**K0, K0); NaN where expert already in set
    optional_experts: List[str]

    @property
    def n_queries(self) -> int:
        return self.value_matrix.shape[0]

    @property
    def n_subsets(self) -> int:
        return self.value_matrix.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.query_feats.shape[1]

    def best_subset_value(self) -> np.ndarray:
        return self.value_matrix.max(axis=1)

    def empty_set_value(self) -> np.ndarray:
        return self.value_matrix[:, 0]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, metric=self.metric, query_feats=self.query_feats,
            value_matrix=self.value_matrix, marginal=self.marginal,
            optional_experts=np.array(self.optional_experts))

    @classmethod
    def load(cls, path: str) -> "OracleLabels":
        d = np.load(path, allow_pickle=True)
        return cls(metric=str(d["metric"]), query_feats=d["query_feats"],
                   value_matrix=d["value_matrix"], marginal=d["marginal"],
                   optional_experts=list(d["optional_experts"]))


def build_oracle_labels(engine: RetrievalEngine,
                        priors: Sequence[QueryExpertPriors],
                        gt_video_ids: Sequence[str],
                        metric: str = "rr",
                        verbose: bool = True) -> OracleLabels:
    Nq = len(gt_video_ids)
    masks = all_optional_masks()
    n_subsets = masks.shape[0]
    experts_per_subset = [mask_to_names(m) for m in masks]

    feat_dim = query_feature_dim(engine.g.clip_dim)
    feats = np.zeros((Nq, feat_dim), dtype=np.float32)
    value_matrix = np.zeros((Nq, n_subsets), dtype=np.float32)

    t0 = time.perf_counter()
    for qi in range(Nq):
        p = priors[qi]
        gt = gt_video_ids[qi]
        sem = engine.semantic_scores(p)
        feats[qi] = extract_query_feature(p, sem)
        for sid in range(n_subsets):
            value_matrix[qi, sid] = engine.value(p, gt, experts_per_subset[sid],
                                                 metric=metric)
        if verbose and (qi + 1) % 50 == 0:
            el = time.perf_counter() - t0
            eta = el / (qi + 1) * (Nq - qi - 1)
            print(f"  [oracle {qi+1}/{Nq}] elapsed={el:.0f}s eta={eta:.0f}s")

    marginal = _compute_marginals(value_matrix, masks)
    return OracleLabels(metric=metric, query_feats=feats,
                        value_matrix=value_matrix, marginal=marginal,
                        optional_experts=list(OPTIONAL_NAMES))


def _compute_marginals(value_matrix: np.ndarray, masks: np.ndarray) -> np.ndarray:
    Nq, n_subsets = value_matrix.shape
    K0 = masks.shape[1]
    marg = np.full((Nq, n_subsets, K0), np.nan, dtype=np.float32)
    for sid in range(n_subsets):
        base = value_matrix[:, sid]
        for j in range(K0):
            if masks[sid, j]:
                continue
            marg[:, sid, j] = value_matrix[:, sid | (1 << j)] - base
    return marg
