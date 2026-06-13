"""Integrated CSER inference pipeline (plan §4.1, §4.5 combined guarantee).

Combines the three modules into one query-time policy:

    1. Greedy Budgeted Selector (Module 3) uses the SVN (Module 1) to pick a
       budget-feasible expert set S from the query feature.
    2. A semantic top-k prefilter proposes a reduced candidate set F(q).
    3. Conformal Safety Gate (Module 2) computes C(q); the final candidate set is
       F(q) ∪ C(q), so no protected video is removed.
    4. Score the retained candidates with the selected experts and rank.

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
    gt_filtered: bool
    cost: float
    n_experts_called: int
    active_experts: List[str]
    gt_in_conformal_set: bool
    conformal_set_size: int
    candidate_count: int
    candidate_reduction_rate: float
    selected_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    fallback_triggered: bool = False
    predicted_best: Optional[float] = None
    predicted_empty: Optional[float] = None

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
                 candidate_top_k: Optional[int] = None,
                 selector=None,
                 safety_mode: str = "reduce",
                 device: str = "cpu") -> None:
        if candidate_top_k is not None and candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive or None")
        if safety_mode not in ("reduce", "report"):
            raise ValueError("safety_mode must be 'reduce' or 'report'")
        self.engine = engine
        if selector is None:
            if model is None:
                raise ValueError("model is required when selector is not supplied")
            selector = GreedyBudgetedSelector(model, budget=budget,
                                             stop_threshold=stop_threshold,
                                             device=device)
        self.selector = selector
        self.gate = conformal_gate
        self.candidate_top_k = candidate_top_k
        self.safety_mode = safety_mode

    def run(self,
            priors: QueryExpertPriors,
            query_feat: np.ndarray,
            gt_video_id: str) -> CSERResult:
        sim_norm = self.engine.semantic_norm(priors)
        gt_idx = self.engine.id_to_idx(gt_video_id)

        sel = self.selector.select(query_feat)

        if self.safety_mode == "report":
            candidate_mask = np.ones(self.engine._N, dtype=bool)
        elif self.candidate_top_k is None:
            candidate_mask = np.ones(self.engine._N, dtype=bool)
        else:
            candidate_mask = self.engine.semantic_top_k_mask(
                sim_norm, self.candidate_top_k)

        if self.gate is not None:
            protected_mask = np.asarray(
                self.gate.prediction_set_mask(sim_norm), dtype=bool)
            if protected_mask.shape != (self.engine._N,):
                raise ValueError(
                    f"protected mask must have shape ({self.engine._N},)")
            if self.safety_mode == "reduce":
                candidate_mask |= protected_mask
            in_set = bool(gt_idx >= 0 and protected_mask[gt_idx])
            set_size = int(protected_mask.sum())
        else:
            in_set, set_size = True, self.engine._N

        gt_filtered = bool(gt_idx >= 0 and not candidate_mask[gt_idx])
        candidate_count = int(candidate_mask.sum())
        rank = self.engine.rank_of_gt(
            priors, gt_video_id, sel.active_experts, candidate_mask=candidate_mask)

        return CSERResult(
            rank=rank, gt_filtered=gt_filtered, cost=sel.cost,
            n_experts_called=sel.n_experts_called,
            active_experts=sel.active_experts,
            gt_in_conformal_set=bool(in_set),
            conformal_set_size=int(set_size),
            candidate_count=candidate_count,
            candidate_reduction_rate=1.0 - candidate_count / self.engine._N,
            selected_mask=sel.selected_mask,
            fallback_triggered=bool(getattr(sel, "fallback_triggered", False)),
            predicted_best=getattr(sel, "predicted_best", None),
            predicted_empty=getattr(sel, "predicted_empty", None),
        )
