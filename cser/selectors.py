"""Inference-time selector variants for CSER Phase 2."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from .experts import (N_OPTIONAL, OPTIONAL_COSTS, OPTIONAL_NAMES,
                      SEMANTIC_COST, all_optional_masks, id_to_mask,
                      mask_to_names)
from .set_value_network import SetValueNetwork
from .svn import SubmodularValueNetwork
from .value_oracle import OracleLabels


SELECTOR_MODES = (
    "marginal_value_greedy",
    "marginal_density_greedy",
    "set_value",
    "set_value_safe",
)

ROSTER_PRESETS = {
    "all": tuple(OPTIONAL_NAMES),
    "no_face_id": tuple(n for n in OPTIONAL_NAMES if n != "face_id"),
    "semantic_highlight_scene": ("highlight", "scene"),
}


@dataclass
class SelectionResult:
    selected_mask: np.ndarray
    active_experts: List[str]
    cost: float
    n_experts_called: int
    trace: List[dict] = field(default_factory=list)
    selector_mode: str = ""
    fallback_triggered: bool = False
    predicted_best: Optional[float] = None
    predicted_empty: Optional[float] = None


def roster_allowed_mask(roster: str = "all") -> np.ndarray:
    """Return a boolean optional-expert mask allowed by a roster preset/list."""
    if roster in ROSTER_PRESETS:
        names = set(ROSTER_PRESETS[roster])
    else:
        names = {x.strip() for x in roster.split(",") if x.strip()}
        unknown = names - set(OPTIONAL_NAMES)
        if unknown:
            raise ValueError(f"unknown expert(s) in roster '{roster}': {sorted(unknown)}")
    return np.array([name in names for name in OPTIONAL_NAMES], dtype=bool)


def _mask_cost(mask: Sequence[bool]) -> float:
    m = np.asarray(mask, dtype=bool)
    return float(SEMANTIC_COST + OPTIONAL_COSTS[m].sum())


def feasible_subset_ids(budget: float,
                        allowed_mask: Optional[np.ndarray] = None) -> List[int]:
    allowed = (np.ones(N_OPTIONAL, dtype=bool) if allowed_mask is None
               else np.asarray(allowed_mask, dtype=bool))
    out = []
    for sid, mask in enumerate(all_optional_masks()):
        if np.any(mask & ~allowed):
            continue
        if _mask_cost(mask) <= budget + 1e-9:
            out.append(int(sid))
    return out


@torch.no_grad()
def calibrate_set_value_min_delta(
        model: SetValueNetwork,
        labels: OracleLabels,
        budget: float,
        roster: str,
        candidates: Sequence[float],
        device: str = "cpu"):
    """Choose the safe-fallback margin using calibration labels only."""
    grid = sorted({float(x) for x in candidates})
    if not grid:
        raise ValueError("min-delta calibration grid must not be empty")
    if any(x < 0 for x in grid):
        raise ValueError("min-delta calibration values must be non-negative")

    allowed = roster_allowed_mask(roster)
    feasible = feasible_subset_ids(budget, allowed)
    if 0 not in feasible:
        raise ValueError("semantic-only subset must be budget feasible")

    model = model.to(device).eval()
    x = torch.from_numpy(labels.query_feats.astype(np.float32)).to(device)
    pred = model(x).cpu().numpy()
    feasible_arr = np.asarray(feasible, dtype=np.int64)
    best_pos = np.argmax(pred[:, feasible_arr], axis=1)
    best_sid = feasible_arr[best_pos]
    row = np.arange(labels.n_queries)
    predicted_best = pred[row, best_sid]
    predicted_empty = pred[:, 0]
    semantic_value = labels.value_matrix[:, 0]

    curve = {}
    for delta in grid:
        fallback = predicted_best <= predicted_empty + delta
        chosen = best_sid.copy()
        chosen[fallback] = 0
        realised = labels.value_matrix[row, chosen]
        curve[f"{delta:.8g}"] = {
            "min_delta": float(delta),
            "mean_value": float(realised.mean()),
            "mean_delta_vs_semantic": float(
                (realised - semantic_value).mean()),
            "degradation_rate": float((realised < semantic_value).mean()),
            "selected_nonempty_rate": float((chosen != 0).mean()),
            "fallback_rate": float(fallback.mean()),
        }

    best = max(
        curve.values(),
        key=lambda item: (
            item["mean_value"],
            -item["degradation_rate"],
            item["min_delta"],
        ),
    )
    return float(best["min_delta"]), {
        "selection_split": "calibration",
        "budget": float(budget),
        "roster": roster,
        "selected_min_delta": float(best["min_delta"]),
        "curve": curve,
    }


class MarginalGreedySelector:
    """Greedy selector over SVN-predicted marginal values."""

    def __init__(self,
                 model: SubmodularValueNetwork,
                 budget: float = 5.0,
                 stop_threshold: float = 0.0,
                 use_density: bool = False,
                 allowed_mask: Optional[np.ndarray] = None,
                 device: str = "cpu") -> None:
        self.model = model.eval()
        self.budget = float(budget)
        self.stop_threshold = float(stop_threshold)
        self.use_density = bool(use_density)
        self.allowed_mask = (np.ones(N_OPTIONAL, dtype=bool) if allowed_mask is None
                             else np.asarray(allowed_mask, dtype=bool))
        self.device = torch.device(device)

    @torch.no_grad()
    def select(self, query_feat: np.ndarray) -> SelectionResult:
        x = torch.from_numpy(np.asarray(query_feat, np.float32)[None, :]).to(self.device)
        selected = np.zeros(N_OPTIONAL, dtype=bool)
        remaining = self.budget - SEMANTIC_COST
        trace: List[dict] = []

        while True:
            mask_t = torch.from_numpy(selected.astype(np.float32)[None, :]).to(self.device)
            pred = self.model(x, mask_t).cpu().numpy().ravel()

            best_j, best_score, best_v = -1, self.stop_threshold, self.stop_threshold
            for j in range(N_OPTIONAL):
                if selected[j] or not self.allowed_mask[j] or OPTIONAL_COSTS[j] > remaining:
                    continue
                v = float(pred[j])
                score = v / max(float(OPTIONAL_COSTS[j]), 1e-9) if self.use_density else v
                if score > best_score:
                    best_j, best_score, best_v = j, score, v

            if best_j < 0:
                break
            selected[best_j] = True
            remaining -= float(OPTIONAL_COSTS[best_j])
            trace.append({
                "added": OPTIONAL_NAMES[best_j],
                "pred_marginal": float(best_v),
                "score": float(best_score),
                "remaining_budget": float(remaining),
            })

        return SelectionResult(
            selected_mask=selected,
            active_experts=mask_to_names(selected),
            cost=_mask_cost(selected),
            n_experts_called=1 + int(selected.sum()),
            trace=trace,
            selector_mode=("marginal_density_greedy" if self.use_density
                           else "marginal_value_greedy"),
        )


class SetValueSelector:
    """Enumerate feasible subsets and select by predicted set value."""

    def __init__(self,
                 model: SetValueNetwork,
                 budget: float = 5.0,
                 min_delta: float = 0.0,
                 safe: bool = False,
                 allowed_mask: Optional[np.ndarray] = None,
                 device: str = "cpu") -> None:
        self.model = model.eval()
        self.budget = float(budget)
        self.min_delta = float(min_delta)
        self.safe = bool(safe)
        self.allowed_mask = (np.ones(N_OPTIONAL, dtype=bool) if allowed_mask is None
                             else np.asarray(allowed_mask, dtype=bool))
        self.device = torch.device(device)
        self._feasible = feasible_subset_ids(self.budget, self.allowed_mask)
        if not self._feasible:
            self._feasible = [0]

    @torch.no_grad()
    def select(self, query_feat: np.ndarray) -> SelectionResult:
        x = torch.from_numpy(np.asarray(query_feat, np.float32)[None, :]).to(self.device)
        pred = self.model(x).cpu().numpy().ravel()
        best_sid = max(self._feasible, key=lambda sid: float(pred[sid]))
        predicted_best = float(pred[best_sid])
        predicted_empty = float(pred[0])
        fallback = False
        if self.safe and predicted_best <= predicted_empty + self.min_delta:
            best_sid = 0
            fallback = True
        mask = id_to_mask(best_sid)
        return SelectionResult(
            selected_mask=mask,
            active_experts=mask_to_names(mask),
            cost=_mask_cost(mask),
            n_experts_called=1 + int(mask.sum()),
            trace=[{
                "selected_subset_id": int(best_sid),
                "predicted_best": predicted_best,
                "predicted_empty": predicted_empty,
                "min_delta": self.min_delta,
                "fallback_triggered": fallback,
            }],
            selector_mode="set_value_safe" if self.safe else "set_value",
            fallback_triggered=fallback,
            predicted_best=predicted_best,
            predicted_empty=predicted_empty,
        )


def load_set_value_model(path: str, d_query: int, d_model: int = 128,
                         device: str = "cpu") -> SetValueNetwork:
    model = SetValueNetwork(d_query=d_query, d_model=d_model, n_experts=N_OPTIONAL)
    state = torch.load(Path(path), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def build_selector(mode: str,
                   budget: float,
                   roster: str = "all",
                   svn_model: Optional[SubmodularValueNetwork] = None,
                   set_value_model: Optional[SetValueNetwork] = None,
                   min_delta: float = 0.0,
                   stop_threshold: float = 0.0,
                   device: str = "cpu"):
    """Factory for the selector modes named in the MSRVTT10K plan."""
    if mode not in SELECTOR_MODES:
        raise ValueError(f"selector must be one of {SELECTOR_MODES}, got '{mode}'")
    allowed = roster_allowed_mask(roster)
    if mode in ("marginal_value_greedy", "marginal_density_greedy"):
        if svn_model is None:
            raise ValueError(f"{mode} requires svn_model")
        return MarginalGreedySelector(
            svn_model, budget=budget, stop_threshold=stop_threshold,
            use_density=(mode == "marginal_density_greedy"),
            allowed_mask=allowed, device=device,
        )
    if set_value_model is None:
        raise ValueError(f"{mode} requires set_value_model")
    return SetValueSelector(
        set_value_model, budget=budget, min_delta=min_delta,
        safe=(mode == "set_value_safe"), allowed_mask=allowed, device=device,
    )
