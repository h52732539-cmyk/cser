"""Greedy Budgeted Selector — Module 3 (plan §4.4).

Given the SVN's predicted marginal values, greedily add the highest-value
feasible expert until the budget is exhausted or the best remaining marginal
falls below a stop threshold. The semantic base (e0) is always implicitly
selected; greedy operates over the K0 optional experts.

This is the inference-time policy whose value the (1-1/e) approximation bound
(plan Theorem 2) refers to. The Conformal Safety Gate's hard constraint will be
layered on top in a later phase; here selection only adds *soft* signals, so no
safety constraint is needed yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .experts import (N_OPTIONAL, OPTIONAL_COSTS, OPTIONAL_NAMES,
                      SEMANTIC_COST, mask_to_names)
from .svn import SubmodularValueNetwork


@dataclass
class GreedyResult:
    selected_mask: np.ndarray              # (K0,) bool, optional experts chosen
    active_experts: List[str]              # names of chosen experts
    cost: float                            # total budget consumed (incl. base)
    n_experts_called: int                  # base + optional
    trace: List[dict] = field(default_factory=list)   # per-step decision log


class GreedyBudgetedSelector:
    def __init__(self,
                 model: SubmodularValueNetwork,
                 budget: float = 3.0,
                 stop_threshold: float = 0.0,
                 device: str = "cpu") -> None:
        self.model = model.eval()
        self.budget = float(budget)
        self.stop_threshold = float(stop_threshold)
        self.device = torch.device(device)

    @torch.no_grad()
    def select(self, query_feat: np.ndarray) -> GreedyResult:
        x = torch.from_numpy(np.asarray(query_feat, np.float32)[None, :]).to(self.device)
        selected = np.zeros(N_OPTIONAL, dtype=bool)
        remaining = self.budget - SEMANTIC_COST       # base is always paid
        trace: List[dict] = []

        while True:
            mask_t = torch.from_numpy(selected.astype(np.float32)[None, :]).to(self.device)
            pred = self.model(x, mask_t).cpu().numpy().ravel()      # (K0,)

            # Feasible = not yet selected and affordable.
            best_j, best_v = -1, self.stop_threshold
            for j in range(N_OPTIONAL):
                if selected[j] or OPTIONAL_COSTS[j] > remaining:
                    continue
                if pred[j] > best_v:
                    best_v, best_j = pred[j], j

            if best_j < 0:                              # nothing worth adding
                break
            selected[best_j] = True
            remaining -= float(OPTIONAL_COSTS[best_j])
            trace.append({"added": OPTIONAL_NAMES[best_j],
                          "pred_marginal": float(best_v),
                          "remaining_budget": float(remaining)})

        cost = float(SEMANTIC_COST + OPTIONAL_COSTS[selected].sum())
        return GreedyResult(
            selected_mask=selected,
            active_experts=mask_to_names(selected),
            cost=cost,
            n_experts_called=1 + int(selected.sum()),
            trace=trace,
        )
