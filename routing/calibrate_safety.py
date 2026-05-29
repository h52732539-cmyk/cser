"""Calibrated safety threshold selection.

For each metadata axis, select the lowest threshold τ such that the
Clopper-Pearson upper confidence bound of the false-elimination rate
is ≤ δ on the calibration split.

This ensures that when C-QIN says "safe to hard-filter on this axis",
the probability of accidentally removing the GT video is controlled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


SAFETY_AXES = ("time", "geo", "motion", "device")


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Upper bound of Clopper-Pearson confidence interval for binomial p.

    Returns the (1-alpha) upper confidence bound for the true failure
    rate given k failures in n trials.
    """
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    from scipy.stats import beta as beta_dist
    return float(beta_dist.ppf(1 - alpha, k + 1, n - k))


def clopper_pearson_upper_fallback(k: int, n: int, alpha: float = 0.05) -> float:
    """Fallback without scipy — Wilson score upper bound."""
    if n == 0:
        return 1.0
    p_hat = k / n
    z = 1.96 if alpha == 0.05 else 1.645
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return min(1.0, center + margin)


def _ucb_fn(k: int, n: int, alpha: float) -> float:
    try:
        return clopper_pearson_upper(k, n, alpha)
    except ImportError:
        return clopper_pearson_upper_fallback(k, n, alpha)


@dataclass
class CalibrationResult:
    axis: str
    tau: float
    enabled: bool
    n_accepted: int
    n_total: int
    empirical_failure_rate: float
    ucb_failure_rate: float


def calibrate_one_axis(
    safety_scores: np.ndarray,
    failure_labels: np.ndarray,
    delta: float = 0.05,
    alpha: float = 0.05,
    min_accept: int = 30,
) -> CalibrationResult:
    """Select threshold for one axis on calibration data.

    Args:
        safety_scores: (N,) sigmoid outputs from C-QIN safety head
        failure_labels: (N,) binary, 1 = GT was eliminated by this axis
        delta: target upper bound on failure rate
        alpha: confidence level for Clopper-Pearson
        min_accept: minimum accepted samples to consider a threshold
    """
    N = len(safety_scores)
    if N < min_accept:
        return CalibrationResult(
            axis="", tau=1.0, enabled=False,
            n_accepted=0, n_total=N,
            empirical_failure_rate=0.0, ucb_failure_rate=1.0,
        )

    thresholds = np.sort(np.unique(safety_scores))

    best = CalibrationResult(
        axis="", tau=1.0, enabled=False,
        n_accepted=0, n_total=N,
        empirical_failure_rate=0.0, ucb_failure_rate=1.0,
    )

    for tau in thresholds:
        accepted = safety_scores >= tau
        n_acc = int(accepted.sum())
        if n_acc < min_accept:
            continue
        k = int(failure_labels[accepted].sum())
        emp_rate = k / n_acc
        ucb = _ucb_fn(k, n_acc, alpha)
        if ucb <= delta:
            best = CalibrationResult(
                axis="", tau=float(tau), enabled=True,
                n_accepted=n_acc, n_total=N,
                empirical_failure_rate=emp_rate,
                ucb_failure_rate=ucb,
            )
            break

    return best


def calibrate_all_axes(
    safety_probs: np.ndarray,
    survival_labels: np.ndarray,
    delta: float = 0.05,
    alpha: float = 0.05,
    min_accept: int = 30,
) -> Dict[str, CalibrationResult]:
    """Calibrate thresholds for all 4 axes.

    Args:
        safety_probs: (N, 4) from C-QIN safety head
        survival_labels: (N, 4) binary, 1 = GT survived
    """
    failure_labels = 1 - survival_labels.astype(np.float32)
    results = {}
    for i, axis in enumerate(SAFETY_AXES):
        r = calibrate_one_axis(
            safety_probs[:, i], failure_labels[:, i],
            delta=delta, alpha=alpha, min_accept=min_accept,
        )
        r.axis = axis
        results[axis] = r
    return results


def save_calibration(results: Dict[str, CalibrationResult],
                      path: str) -> None:
    out = {}
    for axis, r in results.items():
        out[axis] = {
            "tau": r.tau,
            "enabled": r.enabled,
            "n_accepted": r.n_accepted,
            "n_total": r.n_total,
            "empirical_failure_rate": r.empirical_failure_rate,
            "ucb_failure_rate": r.ucb_failure_rate,
        }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


def load_calibration(path: str) -> Dict[str, CalibrationResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for axis, d in data.items():
        out[axis] = CalibrationResult(
            axis=axis, tau=d["tau"], enabled=d["enabled"],
            n_accepted=d["n_accepted"], n_total=d["n_total"],
            empirical_failure_rate=d["empirical_failure_rate"],
            ucb_failure_rate=d["ucb_failure_rate"],
        )
    return out
