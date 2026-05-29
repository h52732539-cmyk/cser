"""Calibrated planner v2 — dual-threshold routing with soft fallback.

Three-zone safety classification per axis:
  - HARD-SAFE:     S_a(q) >= tau_hard  → allow hard filter on this axis
  - SOFT-UNCERTAIN: tau_soft <= S_a(q) < tau_hard → use soft rerank only
  - UNSAFE:        S_a(q) < tau_soft  → do not use this axis at all

This eliminates the "all-or-nothing" fallback that caused MeanR degradation:
  - Old B7: if ANY axis unsafe → fallback to pure semantic (MeanR blows up)
  - New B9: if axis uncertain → degrade to soft rerank (preserves ranking boost)
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


# ----------------------------------------------------------------------
#  Dual-threshold calibration result
# ----------------------------------------------------------------------

@dataclass
class DualThreshold:
    axis: str
    tau_hard: float     # above → hard filter allowed
    tau_soft: float     # above → soft rerank allowed (below → disable axis)
    enabled: bool


def build_dual_thresholds(
    calibration: Dict[str, CalibrationResult],
    soft_ratio: float = 0.6,
) -> Dict[str, DualThreshold]:
    """Derive dual thresholds from single calibration results.

    tau_hard = calibrated tau (existing)
    tau_soft = tau_hard * soft_ratio (more permissive)
    """
    out = {}
    for axis, cr in calibration.items():
        if not cr.enabled:
            out[axis] = DualThreshold(axis=axis, tau_hard=1.0,
                                        tau_soft=1.0, enabled=False)
        else:
            tau_hard = cr.tau
            tau_soft = max(0.0, tau_hard * soft_ratio)
            out[axis] = DualThreshold(
                axis=axis, tau_hard=tau_hard, tau_soft=tau_soft, enabled=True,
            )
    return out


# ----------------------------------------------------------------------
#  Per-axis safety zone
# ----------------------------------------------------------------------

class AxisZone:
    HARD_SAFE = "hard_safe"
    SOFT_UNCERTAIN = "soft_uncertain"
    UNSAFE = "unsafe"


def classify_axis_zone(safety_prob: float, dt: DualThreshold) -> str:
    if not dt.enabled:
        return AxisZone.UNSAFE
    if safety_prob >= dt.tau_hard:
        return AxisZone.HARD_SAFE
    elif safety_prob >= dt.tau_soft:
        return AxisZone.SOFT_UNCERTAIN
    else:
        return AxisZone.UNSAFE


# ----------------------------------------------------------------------
#  Planner decision (extended)
# ----------------------------------------------------------------------

@dataclass
class PlannerDecisionV2:
    selected_route: RetrievalRoute
    route_value: float
    safety_probs: Dict[str, float]
    axis_zones: Dict[str, str]      # axis → zone label
    used_fallback: bool
    fallback_type: str              # "none" / "soft" / "semantic"
    n_hard_safe_routes: int
    n_soft_fallback_routes: int
    n_total_routes: int


# ----------------------------------------------------------------------
#  Soft fallback route construction
# ----------------------------------------------------------------------

def _build_soft_route_for_axes(soft_axes: List[str]) -> RetrievalRoute:
    """Dynamically build a soft-only route for uncertain axes."""
    route_id = f"dynamic_soft_{'_'.join(sorted(soft_axes))}"
    return RetrievalRoute(
        route_id=route_id,
        description=f"Dynamic soft rerank: {soft_axes}",
        hard_axes=(),
        soft_axes=tuple(soft_axes),
        candidate_topm=500,
        rerank_mode="nnn_qamp",
        budget_tier="medium" if len(soft_axes) > 1 else "low",
    )


# ----------------------------------------------------------------------
#  CalibratedPlannerV2
# ----------------------------------------------------------------------

class CalibratedPlannerV2:
    """Dual-threshold planner with soft fallback."""

    def __init__(self,
                 model: CalibratedQIN,
                 bank: RouteBank,
                 calibration: Dict[str, CalibrationResult],
                 soft_ratio: float = 0.6,
                 device: str = "cpu") -> None:
        self.model = model.eval()
        self.bank = bank
        self.dual_thresholds = build_dual_thresholds(calibration, soft_ratio)
        self.device = torch.device(device)
        self.model.to(self.device)
        self._axis_to_idx = {a: i for i, a in enumerate(SAFETY_AXES)}

    def plan(self, features: np.ndarray,
              intent: Optional[QueryIntent] = None) -> PlannerDecisionV2:
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        route_values = out["route_values"][0].cpu().numpy()
        safety_probs = out["safety_probs"][0].cpu().numpy()

        safety_dict = {a: float(safety_probs[i])
                        for i, a in enumerate(SAFETY_AXES)}

        # Classify each axis into zones
        axis_zones = {}
        for axis in SAFETY_AXES:
            dt = self.dual_thresholds.get(axis)
            if dt is None:
                axis_zones[axis] = AxisZone.UNSAFE
            else:
                ax_idx = self._axis_to_idx[axis]
                axis_zones[axis] = classify_axis_zone(
                    safety_probs[ax_idx], dt
                )

        # --- Build unified candidate pool ---
        # Each candidate: (route, value, tier) where tier ∈ {hard, soft, semantic}
        candidates: List[Tuple[RetrievalRoute, float, str]] = []
        n_hard = 0
        n_soft = 0

        for idx, route in enumerate(self.bank):
            val = float(route_values[idx])

            if not route.has_hard_filter and not route.has_soft_rerank:
                # Pure semantic route — always eligible
                candidates.append((route, val, "semantic"))
                continue

            # Check hard_axes: all must be HARD_SAFE
            all_hard_safe = all(
                axis_zones.get(a) == AxisZone.HARD_SAFE
                for a in route.hard_axes
            )
            # Check soft_axes: each must be HARD_SAFE or SOFT_UNCERTAIN
            all_soft_ok = all(
                axis_zones.get(a) in (AxisZone.HARD_SAFE, AxisZone.SOFT_UNCERTAIN)
                for a in route.soft_axes
            )

            if all_hard_safe and all_soft_ok:
                candidates.append((route, val, "hard"))
                n_hard += 1
            elif not route.has_hard_filter and all_soft_ok:
                # No hard filter but has soft axes that are safe enough
                candidates.append((route, val, "soft"))
                n_soft += 1

        # --- Also build dynamic soft routes for uncertain axes ---
        uncertain_axes = [a for a, z in axis_zones.items()
                           if z == AxisZone.SOFT_UNCERTAIN]
        if uncertain_axes:
            dyn_route = _build_soft_route_for_axes(uncertain_axes)
            # Estimate value: mean of similar soft routes in candidates
            soft_vals = [v for _, v, t in candidates if t == "soft"]
            sem_vals = [v for _, v, t in candidates if t == "semantic"]
            # Dynamic soft should be slightly better than pure semantic
            dyn_val = float(np.mean(soft_vals)) if soft_vals else (
                float(np.mean(sem_vals)) * 1.05 if sem_vals else 0.0
            )
            candidates.append((dyn_route, dyn_val, "soft"))
            n_soft += 1

        # --- Pick best candidate ---
        if not candidates:
            selected = self.bank.fallback
            best_val = 0.0
            fallback_type = "semantic"
            used_fallback = True
        else:
            best_route, best_val, best_tier = max(candidates, key=lambda x: x[1])
            selected = best_route
            if best_tier == "semantic" and n_hard == 0 and n_soft == 0:
                fallback_type = "semantic"
                used_fallback = True
            elif best_tier == "soft":
                fallback_type = "soft"
                used_fallback = True
            else:
                fallback_type = "none"
                used_fallback = False

        return PlannerDecisionV2(
            selected_route=selected,
            route_value=best_val,
            safety_probs=safety_dict,
            axis_zones=axis_zones,
            used_fallback=used_fallback,
            fallback_type=fallback_type,
            n_hard_safe_routes=n_hard,
            n_soft_fallback_routes=n_soft,
            n_total_routes=len(self.bank),
        )

    def plan_and_execute(self,
                          features: np.ndarray,
                          query_emb: np.ndarray,
                          gt_video_id: str,
                          intent: Optional[QueryIntent],
                          executor: RouteExecutor,
                          ) -> Tuple[PlannerDecisionV2, RouteResult]:
        decision = self.plan(features, intent)
        result = executor.execute(
            decision.selected_route, query_emb, gt_video_id, intent,
        )
        return decision, result


# ----------------------------------------------------------------------
#  Budgeted cascade planner (B10)
# ----------------------------------------------------------------------

class BudgetedCascadePlanner:
    """B10: C-QIN calibrated + budgeted cascade.

    Stage 1: Use C-QIN to pick best route (hard-safe or soft-fallback).
    Stage 2: If rank > 5, escalate to higher budget route.
    Stage 3: If still > 10, go full budget.
    """

    def __init__(self, planner_v2: CalibratedPlannerV2,
                 bank: RouteBank) -> None:
        self.planner = planner_v2
        self.bank = bank

    def plan_and_execute(self,
                          features: np.ndarray,
                          query_emb: np.ndarray,
                          gt_video_id: str,
                          intent: Optional[QueryIntent],
                          executor: RouteExecutor,
                          ) -> RouteResult:
        # Stage 1: C-QIN route
        decision, res1 = self.planner.plan_and_execute(
            features, query_emb, gt_video_id, intent, executor,
        )
        if res1.rank >= 0 and res1.rank < 5:
            return res1

        # Stage 2: medium budget (top1000 + nnn_qamp)
        r_med = self.bank.get("R16_semantic_top1000_nnn_qamp") or FALLBACK_ROUTE
        res2 = executor.execute(r_med, query_emb, gt_video_id, intent)
        if res2.rank >= 0 and res2.rank < res1.rank:
            res2_better = res2
        else:
            res2_better = res1

        if res2_better.rank >= 0 and res2_better.rank < 10:
            return res2_better

        # Stage 3: full budget
        r_full = self.bank.get("R29_full_budget_all_soft_dense_refine") or FALLBACK_ROUTE
        res3 = executor.execute(r_full, query_emb, gt_video_id, intent)
        # Return best across all stages
        candidates = [res1, res2, res3]
        valid = [r for r in candidates if r.rank >= 0]
        if not valid:
            return res1
        return min(valid, key=lambda r: r.rank)
