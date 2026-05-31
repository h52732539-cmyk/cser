"""Module 2 — Conformal Safety Gate (plan §4.3).

Produces, for any query q, a prediction set C(q) ⊆ gallery with the
distribution-free guarantee

    P( v* ∈ C(q) ) ≥ 1 - α

requiring only exchangeability of calibration and test queries (split conformal,
Vovk et al.). C(q) is the set the routing layer is forbidden to filter out, so
its coverage is exactly "the correct video is never dropped" at level 1-α.

Nonconformity score (plan §4.3, simplified to the signals we actually have):

    s(q, v) = 1 - sim_norm(q, v)

where ``sim_norm`` is the per-query min-max-normalised semantic similarity in
[0, 1] (``RetrievalEngine.semantic_norm``). A low score = high similarity = the
video conforms. We calibrate the GT scores: the GT should have a *small*
nonconformity score, so the threshold q̂ is an upper quantile of calibration GT
scores, and C(q) = { v : s(q,v) ≤ q̂ }.

Two calibrators:

* :class:`SplitConformal`     — one global threshold (Theorem 1).
* :class:`MondrianConformal`  — per-difficulty-bin thresholds (plan "adaptive"
  extension): tighter sets for easy queries, valid coverage within each bin.

Difficulty is the QPP margin (top1 - top2 of the normalised semantic scores):
large margin = easy query.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-α) quantile of calibration scores.

    Uses the split-conformal level ⌈(n+1)(1-α)⌉ / n (Vovk). Returns +inf when
    the corrected rank exceeds n, i.e. C(q) must include everything to keep the
    guarantee with so few calibration points.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def gt_nonconformity(sim_norm: np.ndarray, gt_idx: int) -> float:
    """Nonconformity score of the GT video for one query: 1 - sim_norm(q, v*)."""
    if gt_idx < 0:
        return 1.0
    return float(1.0 - sim_norm[gt_idx])


# ----------------------------------------------------------------------
#  Difficulty binning (Mondrian taxonomy)
# ----------------------------------------------------------------------

def qpp_margin(sim_norm: np.ndarray) -> float:
    """Easy-query indicator: gap between the two highest normalised sims."""
    if sim_norm.size < 2:
        return 0.0
    top2 = np.partition(sim_norm, -2)[-2:]
    return float(abs(top2[1] - top2[0]))


def difficulty_bin(margin: float, edges: Sequence[float]) -> int:
    """Map a margin to a bin index given sorted interior edges."""
    b = 0
    for e in edges:
        if margin >= e:
            b += 1
    return b


# ----------------------------------------------------------------------
#  Split conformal (global threshold)
# ----------------------------------------------------------------------

@dataclass
class SplitConformal:
    alpha: float
    threshold: float
    n_calib: int

    @classmethod
    def calibrate(cls, gt_scores: np.ndarray, alpha: float) -> "SplitConformal":
        return cls(alpha=alpha,
                   threshold=conformal_quantile(gt_scores, alpha),
                   n_calib=len(gt_scores))

    def contains(self, sim_norm: np.ndarray, video_idx: int) -> bool:
        """Is video ``video_idx`` in C(q)?  (s ≤ threshold)"""
        return bool((1.0 - sim_norm[video_idx]) <= self.threshold)

    def prediction_set_mask(self, sim_norm: np.ndarray) -> np.ndarray:
        """Boolean (N,) mask of C(q) over the gallery."""
        return (1.0 - sim_norm) <= self.threshold

    def set_size(self, sim_norm: np.ndarray) -> int:
        return int(self.prediction_set_mask(sim_norm).sum())

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["kind"] = "split"
        return d


# ----------------------------------------------------------------------
#  Mondrian conformal (per-difficulty-bin thresholds)
# ----------------------------------------------------------------------

@dataclass
class MondrianConformal:
    alpha: float
    bin_edges: List[float]                 # interior margin edges
    thresholds: List[float]                # one per bin
    n_calib_per_bin: List[int]

    @classmethod
    def calibrate(cls, gt_scores: np.ndarray, margins: np.ndarray,
                  alpha: float, n_bins: int = 3) -> "MondrianConformal":
        """Quantile-binned by margin; per-bin conformal threshold."""
        margins = np.asarray(margins, dtype=np.float64)
        # Interior edges from margin quantiles -> roughly equal-mass bins.
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = list(np.quantile(margins, qs)) if n_bins > 1 else []
        thr, counts = [], []
        for b in range(n_bins):
            in_bin = np.array([difficulty_bin(m, edges) == b for m in margins])
            sc = gt_scores[in_bin]
            thr.append(conformal_quantile(sc, alpha))
            counts.append(int(in_bin.sum()))
        return cls(alpha=alpha, bin_edges=edges, thresholds=thr,
                   n_calib_per_bin=counts)

    def _bin_for(self, sim_norm: np.ndarray) -> int:
        return difficulty_bin(qpp_margin(sim_norm), self.bin_edges)

    def threshold_for(self, sim_norm: np.ndarray) -> float:
        return self.thresholds[self._bin_for(sim_norm)]

    def contains(self, sim_norm: np.ndarray, video_idx: int) -> bool:
        return bool((1.0 - sim_norm[video_idx]) <= self.threshold_for(sim_norm))

    def prediction_set_mask(self, sim_norm: np.ndarray) -> np.ndarray:
        return (1.0 - sim_norm) <= self.threshold_for(sim_norm)

    def set_size(self, sim_norm: np.ndarray) -> int:
        return int(self.prediction_set_mask(sim_norm).sum())

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["kind"] = "mondrian"
        return d


# ----------------------------------------------------------------------
#  Coverage validation (experiment E3)
# ----------------------------------------------------------------------

@dataclass
class CoverageReport:
    alpha: float
    target_coverage: float
    empirical_coverage: float
    avg_set_size: float
    median_set_size: float
    n_test: int
    kind: str

    def to_dict(self) -> Dict:
        return asdict(self)


def evaluate_coverage(gate,
                      sim_norms: Sequence[np.ndarray],
                      gt_indices: Sequence[int],
                      kind: str = "") -> CoverageReport:
    """Empirical coverage + set size of a calibrated gate on a test split.

    ``sim_norms[i]`` is the (N,) normalised-similarity vector for test query i;
    ``gt_indices[i]`` is the gallery index of its ground-truth video.
    """
    covered, sizes = [], []
    for sn, gi in zip(sim_norms, gt_indices):
        if gi < 0:
            continue
        covered.append(gate.contains(sn, gi))
        sizes.append(gate.set_size(sn))
    cov = float(np.mean(covered)) if covered else 0.0
    return CoverageReport(
        alpha=gate.alpha,
        target_coverage=1.0 - gate.alpha,
        empirical_coverage=cov,
        avg_set_size=float(np.mean(sizes)) if sizes else 0.0,
        median_set_size=float(np.median(sizes)) if sizes else 0.0,
        n_test=len(covered),
        kind=kind or getattr(gate, "to_dict", lambda: {})().get("kind", ""),
    )


def save_gate(gate, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")
