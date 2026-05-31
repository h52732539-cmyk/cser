"""Submodularity verification — experiment E2 (plan §7-E2, §11 fallback).

This is the go/no-go analysis for the whole CSER story. Using the *exact* oracle
value lattice (every f(S, q) enumerated), we measure on real data whether the
value function actually exhibits diminishing returns.

We report three things the paper needs:

1. **Monotonicity violation rate** — fraction of (S, e) pairs where adding an
   expert *decreases* value. (Soft-signal value functions should be ~monotone.)

2. **Submodularity violation rate** — fraction of pairs (S ⊆ S', e ∉ S') where
   the marginal grows with the set:  f(S'∪e)-f(S') > f(S∪e)-f(S) + tol.
   The plan's success criterion is < 5%; the §11 risk trigger is > 10%.

3. **Submodularity ratio γ** (weak-submodular fallback). The greedy guarantee
   degrades gracefully to (1 - e^{-γ}); γ ∈ (0, 1], γ=1 ⇔ fully submodular.
   We estimate γ from the lattice via the standard ratio of summed singleton
   marginals to set gains (Das & Kempe 2011 style, adapted to this lattice).

If submodularity holds (violation < 10%, γ close to 1) → keep the (1-1/e) story.
If not → the paper pivots to the γ-weakly-submodular bound, which these numbers
directly supply.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

from .experts import N_OPTIONAL, all_optional_masks, mask_to_id
from .value_oracle import OracleLabels


@dataclass
class SubmodularityReport:
    n_queries: int
    metric: str
    monotonicity_violation_rate: float
    submodularity_violation_rate: float
    submodularity_violation_pairs: int
    submodularity_total_pairs: int
    mean_violation_magnitude: float          # avg positive excess among violations
    gamma_ratio_mean: float                  # weak-submodularity ratio (mean)
    gamma_ratio_p10: float                   # 10th percentile (conservative bound)
    verdict: str                             # "submodular" | "weakly_submodular" | "non_submodular"

    def to_dict(self) -> Dict:
        return asdict(self)


def _all_superset_pairs() -> List[tuple]:
    """Enumerate (sid_S, sid_Sp, j) with S ⊂ S', j ∉ S'.

    Restricted to S' = S ∪ {one extra expert} (immediate supersets), which is
    sufficient to certify/refute submodularity over the whole lattice.
    """
    masks = all_optional_masks()
    pairs = []
    for sid_s in range(len(masks)):
        s = masks[sid_s]
        for k in range(N_OPTIONAL):                  # expert k grows S -> S'
            if s[k]:
                continue
            sid_sp = sid_s | (1 << k)
            for j in range(N_OPTIONAL):              # candidate marginal expert
                if j == k or masks[sid_sp][j]:
                    continue
                pairs.append((sid_s, sid_sp, j))
    return pairs


def verify_submodularity(labels: OracleLabels,
                         tol: float = 1e-4) -> SubmodularityReport:
    V = labels.value_matrix                          # (Nq, 2**K0)
    Nq = labels.n_queries
    marg = labels.marginal                           # (Nq, 2**K0, K0)

    # --- monotonicity: marginal should be >= 0 ---
    valid = ~np.isnan(marg)
    mono_viol = (marg < -tol) & valid
    mono_rate = float(mono_viol.sum() / max(valid.sum(), 1))

    # --- submodularity: marginal must not increase as the set grows ---
    pairs = _all_superset_pairs()
    n_viol = 0
    n_total = 0
    excess_sum = 0.0
    for (sid_s, sid_sp, j) in pairs:
        m_small = marg[:, sid_s, j]                  # f(S∪j) - f(S)
        m_big = marg[:, sid_sp, j]                   # f(S'∪j) - f(S')
        excess = m_big - m_small                     # > 0  => violation
        n_total += Nq
        v = excess > tol
        n_viol += int(v.sum())
        excess_sum += float(excess[v].sum()) if v.any() else 0.0
    sub_rate = n_viol / max(n_total, 1)
    mean_excess = excess_sum / max(n_viol, 1)

    gamma_mean, gamma_p10 = _submodularity_ratio(V, marg)

    if sub_rate < 0.05:
        verdict = "submodular"
    elif sub_rate <= 0.10:
        verdict = "weakly_submodular"
    else:
        verdict = "non_submodular"

    return SubmodularityReport(
        n_queries=Nq, metric=labels.metric,
        monotonicity_violation_rate=mono_rate,
        submodularity_violation_rate=sub_rate,
        submodularity_violation_pairs=n_viol,
        submodularity_total_pairs=n_total,
        mean_violation_magnitude=mean_excess,
        gamma_ratio_mean=gamma_mean,
        gamma_ratio_p10=gamma_p10,
        verdict=verdict,
    )


def _submodularity_ratio(V: np.ndarray, marg: np.ndarray):
    """Estimate the submodularity ratio γ per query.

    γ for a set S = [sum_{e∈S} f({e})-f(∅)] / [f(S) - f(∅)], clipped to (0, 1].
    Averaged over non-empty subsets; γ≈1 ⇔ submodular, smaller ⇔ more
    super-modular interaction. Returns (mean, 10th-percentile) across queries.
    """
    Nq = V.shape[0]
    masks = all_optional_masks()
    f_empty = V[:, 0]
    # Singleton gains f({e}) - f(∅) for each expert.
    singleton_gain = np.zeros((Nq, N_OPTIONAL), dtype=np.float64)
    for e in range(N_OPTIONAL):
        sid_e = 1 << e
        singleton_gain[:, e] = V[:, sid_e] - f_empty

    per_query = []
    for q in range(Nq):
        ratios = []
        for sid in range(1, len(masks)):
            members = [e for e in range(N_OPTIONAL) if masks[sid][e]]
            if len(members) < 2:
                continue
            set_gain = V[q, sid] - f_empty[q]
            sum_singletons = float(singleton_gain[q, members].sum())
            if set_gain <= 1e-9:
                continue                             # no gain -> ratio undefined
            r = sum_singletons / set_gain
            ratios.append(min(max(r, 0.0), 1.0))
        if ratios:
            per_query.append(float(np.mean(ratios)))
    if not per_query:
        return 1.0, 1.0
    arr = np.array(per_query)
    return float(arr.mean()), float(np.percentile(arr, 10))
