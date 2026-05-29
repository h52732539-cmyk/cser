"""Baseline expert-subset policies for CSER experiments."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .schema import DEFAULT_EXPERTS
from .subset_executor import CSERSubsetExecutor


def semantic_only(executor: CSERSubsetExecutor, query_emb, gt_video_id, query_context=None):
    return executor.execute_subset(("clip_semantic",), query_emb, gt_video_id, query_context)


def all_experts(executor: CSERSubsetExecutor, query_emb, gt_video_id, query_context=None):
    return executor.execute_subset(
        tuple(spec.expert_id for spec in DEFAULT_EXPERTS),
        query_emb,
        gt_video_id,
        query_context,
    )


def fixed_cascade_subset(budget: float) -> tuple[str, ...]:
    order = ("clip_semantic", "face_detect", "scene", "highlight", "arcface")
    selected = []
    used = 0.0
    for expert_id in order:
        if used + 1.0 <= budget + 1e-8:
            selected.append(expert_id)
            used += 1.0
    return tuple(selected)


class RandomSubsetPolicy:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, budget: float) -> tuple[str, ...]:
        optional = ["face_detect", "arcface", "highlight", "scene"]
        self.rng.shuffle(optional)
        n_extra = max(0, int(round(budget)) - 1)
        return tuple(["clip_semantic"] + optional[:n_extra])
