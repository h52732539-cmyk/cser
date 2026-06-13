"""Empirical verification of the three theorems (plan §4.5, §5).

The formal proofs live in docs/delivery/CSER_THEOREMS.md. This module checks that
the *bounds the theorems promise* actually hold on data — the "verify all
theoretical claims empirically" step of the plan (§10 Week 7).

* Theorem 1 (conformal coverage): empirical P(v* ∈ C(q)) ≥ 1 - α.
* Theorem 2 (greedy approximation): f(S_greedy) ≥ (1 - e^{-γ})·f(S*) - K·ε,
  with ε = measured max |v̂ - v*| (SVN surrogate error) and γ the submodularity
  ratio. We report the realised LHS, the bound RHS, and whether LHS ≥ RHS.
* Theorem 3 (combined): all of {coverage, budget, near-optimality} hold at once.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .experts import (N_OPTIONAL, OPTIONAL_COSTS, SEMANTIC_COST,
                      all_optional_masks, mask_to_id)
from .value_oracle import OracleLabels
from .svn import SubmodularValueNetwork
from .greedy import GreedyBudgetedSelector


# ----------------------------------------------------------------------
#  Theorem 1 — conformal coverage
# ----------------------------------------------------------------------

def verify_theorem1_coverage(gate, sim_norms: Sequence[np.ndarray],
                             gt_indices: Sequence[int]) -> Dict:
    covered = [gate.contains(sn, gi) for sn, gi in zip(sim_norms, gt_indices)
               if gi >= 0]
    emp = float(np.mean(covered)) if covered else 0.0
    target = 1.0 - gate.alpha
    return {
        "alpha": gate.alpha,
        "target_coverage": target,
        "empirical_coverage": emp,
        "holds": bool(emp >= target - 1e-9),
        "n_test": len(covered),
    }


# ----------------------------------------------------------------------
#  Theorem 2 — greedy approximation with learned surrogate
# ----------------------------------------------------------------------

def measure_surrogate_error(model: SubmodularValueNetwork,
                            oracle: OracleLabels) -> float:
    """ε = max over (query, set S, expert e∉S) of |v̂(e|S,q) - v*(e|S,q)|.

    Uses the full lattice of conditioning sets so ε bounds the worst-case
    surrogate error that enters the Theorem-2 bound.
    """
    masks = all_optional_masks().astype(np.float32)
    model.eval()
    max_err = 0.0
    with torch.no_grad():
        feats = torch.from_numpy(oracle.query_feats.astype(np.float32))
        for sid in range(len(masks)):
            m = torch.from_numpy(np.tile(masks[sid], (oracle.n_queries, 1)))
            pred = model(feats, m).numpy()                  # (Nq, K0)
            true = oracle.marginal[:, sid, :]               # (Nq, K0)
            valid = ~np.isnan(true)
            if valid.any():
                err = np.abs(pred[valid] - true[valid])
                max_err = max(max_err, float(err.max()))
    return max_err


def verify_theorem2_greedy(model, oracle: OracleLabels, gamma: float,
                           budget: float = 3.0,
                           monotonicity_violation_rate: Optional[float] = None) -> Dict:
    """Check f(S_greedy) ≥ (1 - e^{-γ})·f(S*) - K·ε per query (mean form)."""
    eps = measure_surrogate_error(model, oracle)
    K = N_OPTIONAL
    approx_factor = 1.0 - np.exp(-max(gamma, 1e-6))

    best = oracle.best_subset_value()                       # f(S*)
    sel = GreedyBudgetedSelector(model, budget=budget)
    realised = np.array([
        oracle.value_matrix[q, mask_to_id(sel.select(oracle.query_feats[q]).selected_mask)]
        for q in range(oracle.n_queries)
    ])
    lhs = float(realised.mean())
    rhs = float(approx_factor * best.mean() - K * eps)
    return {
        "surrogate_error_eps": eps,
        "submodularity_ratio_gamma": gamma,
        "approx_factor_(1-e^-gamma)": float(approx_factor),
        "K": K,
        "realised_value_LHS": lhs,
        "bound_RHS": rhs,
        "bound_is_non_vacuous": bool(rhs > 0.0),
        "monotonicity_violation_rate": monotonicity_violation_rate,
        "monotonicity_assumption_holds": bool(
            monotonicity_violation_rate is not None
            and monotonicity_violation_rate <= 1e-4),
        "oracle_value": float(best.mean()),
        "realised_pct_of_oracle": float(lhs / max(best.mean(), 1e-9)),
        "bound_holds": bool(lhs >= rhs - 1e-9),
    }


# ----------------------------------------------------------------------
#  Theorem 3 — combined guarantee
# ----------------------------------------------------------------------

def verify_theorem3_combined(thm1: Dict, thm2: Dict,
                             max_observed_cost: float, budget: float) -> Dict:
    budget_ok = bool(max_observed_cost <= budget + 1e-6)
    near_optimality_ok = bool(
        thm2["bound_holds"]
        and thm2.get("bound_is_non_vacuous", False)
        and thm2.get("monotonicity_assumption_holds", False)
    )
    return {
        "coverage_holds": thm1["holds"],
        "near_optimality_holds": near_optimality_ok,
        "near_optimality_bound_holds": thm2["bound_holds"],
        "near_optimality_bound_is_non_vacuous": thm2.get("bound_is_non_vacuous", False),
        "monotonicity_assumption_holds": thm2.get("monotonicity_assumption_holds", False),
        "budget_compliance_holds": budget_ok,
        "max_observed_cost": max_observed_cost,
        "budget": budget,
        "all_three_hold": bool(thm1["holds"] and near_optimality_ok and budget_ok),
    }
