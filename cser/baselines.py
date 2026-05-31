"""Expert-selection baselines for E1 (plan §7 E1 table).

Each baseline is a *selection policy*: given a query it returns a boolean mask
over the optional experts. The same RetrievalEngine then scores and ranks, so
methods differ only in *which experts they choose*, never in the retrieval code
— a fair comparison.

Baselines
---------
* :class:`AllExperts`     — B0: always select all optional experts (full budget).
* :class:`RandomSelect`   — B1: random budget-feasible subset.
* :class:`FixedCascade`   — B2: fixed easy->hard order until budget exhausted.
* :class:`UCBBandit`      — B4: per-expert UCB bandit, reward = realised RR gain,
                            learned online over the query stream.
* :func:`oracle_mask`     — B-oracle: best subset per query from the value matrix.

CSER itself (B6) is :class:`cser.pipeline.CSERPipeline`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .experts import (N_OPTIONAL, OPTIONAL_COSTS, SEMANTIC_COST,
                      all_optional_masks, mask_to_id, id_to_mask)


def _feasible_subset_ids(budget: float) -> List[int]:
    """All subset ids whose total cost ≤ budget."""
    masks = all_optional_masks()
    out = []
    for sid in range(len(masks)):
        cost = SEMANTIC_COST + OPTIONAL_COSTS[masks[sid]].sum()
        if cost <= budget + 1e-9:
            out.append(sid)
    return out


@dataclass
class AllExperts:
    """B0: select every optional expert (subject to budget)."""
    budget: float = float("inf")

    def select(self, query_feat: np.ndarray) -> np.ndarray:
        mask = np.ones(N_OPTIONAL, dtype=bool)
        if SEMANTIC_COST + OPTIONAL_COSTS.sum() <= self.budget + 1e-9:
            return mask
        # fall back to the most-expensive feasible subset
        feas = _feasible_subset_ids(self.budget)
        best = max(feas, key=lambda s: OPTIONAL_COSTS[id_to_mask(s)].sum())
        return id_to_mask(best)


@dataclass
class RandomSelect:
    """B1: uniformly random budget-feasible subset."""
    budget: float = 3.0
    seed: int = 42

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._feasible = _feasible_subset_ids(self.budget)

    def select(self, query_feat: np.ndarray) -> np.ndarray:
        sid = int(self._rng.choice(self._feasible))
        return id_to_mask(sid)


@dataclass
class FixedCascade:
    """B2: add experts in a fixed priority order until budget is spent."""
    budget: float = 3.0
    order: tuple = (0, 1, 2, 3)            # time, geo, motion, device

    def select(self, query_feat: np.ndarray) -> np.ndarray:
        mask = np.zeros(N_OPTIONAL, dtype=bool)
        remaining = self.budget - SEMANTIC_COST
        for j in self.order:
            if OPTIONAL_COSTS[j] <= remaining:
                mask[j] = True
                remaining -= float(OPTIONAL_COSTS[j])
        return mask


class UCBBandit:
    """B4: per-expert UCB1 bandit over the query stream.

    Treats each optional expert as an arm; the reward when an expert is included
    is the realised RR. The caller must call :meth:`update` after observing the
    outcome so the bandit learns online (standard bandit-baseline protocol).
    """

    def __init__(self, budget: float = 3.0, c: float = 1.0, seed: int = 42):
        self.budget = budget
        self.c = c
        self._rng = np.random.default_rng(seed)
        self.counts = np.zeros(N_OPTIONAL, dtype=np.float64)
        self.values = np.zeros(N_OPTIONAL, dtype=np.float64)   # mean reward
        self.t = 0
        self._last_mask: Optional[np.ndarray] = None

    def _ucb(self) -> np.ndarray:
        bonus = np.where(
            self.counts > 0,
            self.c * np.sqrt(np.log(max(self.t, 1) + 1) / np.maximum(self.counts, 1)),
            1e6,                                                # force exploration
        )
        return self.values + bonus

    def select(self, query_feat: np.ndarray) -> np.ndarray:
        self.t += 1
        scores = self._ucb()
        mask = np.zeros(N_OPTIONAL, dtype=bool)
        remaining = self.budget - SEMANTIC_COST
        for j in np.argsort(-scores):
            if OPTIONAL_COSTS[j] <= remaining:
                mask[j] = True
                remaining -= float(OPTIONAL_COSTS[j])
        self._last_mask = mask
        return mask

    def update(self, reward: float) -> None:
        if self._last_mask is None:
            return
        for j in range(N_OPTIONAL):
            if self._last_mask[j]:
                self.counts[j] += 1
                n = self.counts[j]
                self.values[j] += (reward - self.values[j]) / n
        self._last_mask = None


def oracle_mask(value_row: np.ndarray, budget: float) -> np.ndarray:
    """B-oracle: best budget-feasible subset for one query from its value row.

    ``value_row`` is row q of the oracle value matrix (length 2**K0).
    """
    feas = _feasible_subset_ids(budget)
    best = max(feas, key=lambda s: value_row[s])
    return id_to_mask(best)
