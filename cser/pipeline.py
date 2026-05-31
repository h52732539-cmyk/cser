"""Integrated CSER inference pipeline (plan §4.1, §4.5 combined guarantee).

Combines the three modules into one query-time policy:

    1. Greedy Budgeted Selector (Module 3) uses the SVN (Module 1) to pick a
       budget-feasible expert set S from the query feature.
    2. Conformal Safety Gate (Module 2) computes C(q); the final ranking retains
       every video in C(q) (selection adds reranking signals only, never filters,
       so C(q) is retained by construction — coverage is still measured).
    3. Score the gallery with the selected experts and rank.

Returns per-query :class:`CSERResult` with rank, cost, experts used, and whether
the GT fell inside the conformal set — everything E1/E4/E6 need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .expert_features import QueryExpertPriors
from .greedy import GreedyBudgetedSelector
from .retrieval import RetrievalEngine
from .svn import SubmodularValueNetwork


@dataclass
class CSERResult:
    rank: int
    gt_filtered: bool                # always False (soft-only selection)
    cost: float
    n_experts_called: int
    active_experts: List[str]
    gt_in_conformal_set: bool
    conformal_set_size: int
    selected_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def rr(self) -> float:
        return 0.0 if self.rank < 0 else 1.0 / (self.rank + 1.0)


class CSERPipeline:
    def __init__(self,
                 engine: RetrievalEngine,
                 model: SubmodularValueNetwork,
                 conformal_gate=None,
                 budget: float = 5.0,
                 stop_threshold: float = 0.0,
                 device: str = "cpu") -> None:
        self.engine = engine
        self.selector = GreedyBudgetedSelector(model, budget=budget,
                                               stop_threshold=stop_threshold,
                                               device=device)
        self.gate = conformal_gate

    def run(self,
            priors: QueryExpertPriors,
            query_feat: np.ndarray,
            gt_video_id: str) -> CSERResult:
        sim_norm = self.engine.semantic_norm(priors)
        gt_idx = self.engine.id_to_idx(gt_video_id)

        sel = self.selector.select(query_feat)

        if self.gate is not None and gt_idx >= 0:
            in_set = self.gate.contains(sim_norm, gt_idx)
            set_size = self.gate.set_size(sim_norm)
        else:
            in_set, set_size = True, self.engine._N

        rank = self.engine.rank_of_gt(priors, gt_video_id, sel.active_experts)

        return CSERResult(
            rank=rank, gt_filtered=False, cost=sel.cost,
            n_experts_called=sel.n_experts_called,
            active_experts=sel.active_experts,
            gt_in_conformal_set=bool(in_set),
            conformal_set_size=int(set_size),
            selected_mask=sel.selected_mask,
        )
