"""Query feature extraction for CSER."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .schema import DEFAULT_EXPERT_IDS
from .subset_executor import normalize_scores


def qpp_stats(scores: np.ndarray, top_k: int = 20) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(6, dtype=np.float32)
    order = np.sort(arr)[::-1]
    top = order[:top_k]
    if top.size < top_k:
        top = np.pad(top, (0, top_k - top.size), constant_values=float(top.min()))
    norm = normalize_scores(top)
    top1 = float(norm[0])
    top2 = float(norm[1]) if norm.size > 1 else 0.0
    margin = top1 - top2
    probs = norm / (float(norm.sum()) + 1e-9)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    std = float(norm.std())
    concentration = top1 / (float(norm.sum()) + 1e-9)
    return np.asarray([top1, top2, margin, entropy, std, concentration], dtype=np.float32)


def context_indicators(query_context: Optional[Mapping[str, object]]) -> np.ndarray:
    ctx = query_context or {}
    vals = [
        bool(ctx.get("requires_face", False)),
        ctx.get("target_face_embedding") is not None,
        bool(ctx.get("wants_highlight", False)),
        ctx.get("scene_label") is not None,
        bool(ctx.get("requires_scene_filter", False)),
        float(ctx.get("face_threshold", 0.0)) > 0,
        bool(ctx.get("hard_query", False)),
        bool(ctx.get("easy_query", False)),
    ]
    return np.asarray([float(v) for v in vals], dtype=np.float32)


def budget_one_hot(budget: int | float, max_budget: int = 5) -> np.ndarray:
    out = np.zeros(max_budget, dtype=np.float32)
    idx = int(max(1, min(max_budget, round(float(budget))))) - 1
    out[idx] = 1.0
    return out


def build_query_features(
    query_emb: np.ndarray,
    semantic_scores: np.ndarray,
    query_context: Optional[Mapping[str, object]] = None,
    expert_availability: Optional[Sequence[float]] = None,
    budget: int | float = 5,
    max_budget: int = 5,
    query_dim: int = 512,
) -> np.ndarray:
    emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)
    if emb.size < query_dim:
        emb = np.pad(emb, (0, query_dim - emb.size))
    emb = emb[:query_dim]
    emb = emb / (np.linalg.norm(emb) + 1e-9)

    avail = np.ones(len(DEFAULT_EXPERT_IDS), dtype=np.float32)
    if expert_availability is not None:
        arr = np.asarray(expert_availability, dtype=np.float32).reshape(-1)
        avail[: min(arr.size, avail.size)] = arr[: avail.size]

    return np.concatenate(
        [
            emb.astype(np.float32),
            qpp_stats(semantic_scores),
            context_indicators(query_context),
            avail,
            budget_one_hot(budget, max_budget=max_budget),
        ]
    ).astype(np.float32)


def stack_query_features(
    query_embs: np.ndarray,
    semantic_score_rows: np.ndarray,
    query_contexts: Optional[Sequence[Mapping[str, object]]] = None,
    budget: int | float = 5,
) -> np.ndarray:
    rows = []
    query_contexts = query_contexts or [{} for _ in range(len(query_embs))]
    for emb, scores, ctx in zip(query_embs, semantic_score_rows, query_contexts):
        rows.append(build_query_features(emb, scores, ctx, budget=budget))
    return np.stack(rows).astype(np.float32)
