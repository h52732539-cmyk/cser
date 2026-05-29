"""Calibrated planner — inference-time route selection with safety gating.

Flow:
  1. Extract features from query
  2. Run C-QIN → route_values + safety_probs
  3. For each candidate route:
       - Check all hard_axes have safety_prob ≥ calibrated tau
       - If any axis fails → route is "unsafe"
  4. Among safe routes: select argmax route_value
  5. If no route is safe → fallback to semantic-only
  6. Execute the selected route
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .qin_model import CalibratedQIN
from .route_schema import RetrievalRoute, FALLBACK_ROUTE
from .route_bank import RouteBank
from .route_executor import RouteExecutor, RouteResult
from .calibrate_safety import CalibrationResult, SAFETY_AXES

import sys
from pathlib import Path
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from core.query_parser import QueryIntent


@dataclass
class PlannerDecision:
    selected_route: RetrievalRoute
    route_value: float
    safety_probs: Dict[str, float]
    used_fallback: bool
    n_safe_routes: int
    n_total_routes: int


class CalibratedPlanner:
    """Inference-time planner with calibrated safety gating."""

    def __init__(self,
                 model: CalibratedQIN,
                 bank: RouteBank,
                 calibration: Dict[str, CalibrationResult],
                 device: str = "cpu") -> None:
        self.model = model.eval()
        self.bank = bank
        self.calibration = calibration
        self.device = torch.device(device)
        self.model.to(self.device)

        self._axis_to_idx = {a: i for i, a in enumerate(SAFETY_AXES)}

    # ------------------------------------------------------------------

    def plan(self, features: np.ndarray,
              intent: Optional[QueryIntent] = None) -> PlannerDecision:
        """Select the best safe route given query features."""
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        route_values = out["route_values"][0].cpu().numpy()
        safety_probs = out["safety_probs"][0].cpu().numpy()

        safety_dict = {a: float(safety_probs[i])
                        for i, a in enumerate(SAFETY_AXES)}

        # Determine which routes are safe
        safe_routes: List[Tuple[int, float]] = []
        for idx, route in enumerate(self.bank):
            is_safe = True
            for axis in route.hard_axes:
                ax_idx = self._axis_to_idx.get(axis)
                if ax_idx is None:
                    continue
                cal = self.calibration.get(axis)
                if cal is None or not cal.enabled:
                    is_safe = False
                    break
                if safety_probs[ax_idx] < cal.tau:
                    is_safe = False
                    break
            if is_safe:
                safe_routes.append((idx, float(route_values[idx])))

        # Select best safe route or fallback
        if safe_routes:
            best_idx, best_val = max(safe_routes, key=lambda x: x[1])
            selected = self.bank.routes[best_idx]
            used_fallback = False
        else:
            selected = self.bank.fallback
            best_val = float(route_values[self.bank.index_of(selected.route_id)])
            used_fallback = True

        return PlannerDecision(
            selected_route=selected,
            route_value=best_val,
            safety_probs=safety_dict,
            used_fallback=used_fallback,
            n_safe_routes=len(safe_routes),
            n_total_routes=len(self.bank),
        )

    # ------------------------------------------------------------------

    def plan_and_execute(self,
                          features: np.ndarray,
                          query_emb: np.ndarray,
                          gt_video_id: str,
                          intent: Optional[QueryIntent],
                          executor: RouteExecutor) -> Tuple[PlannerDecision, RouteResult]:
        """Plan + execute in one call (convenience for evaluation)."""
        decision = self.plan(features, intent)
        result = executor.execute(
            decision.selected_route, query_emb, gt_video_id, intent,
        )
        return decision, result
