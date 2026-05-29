"""Metrics helpers for CSER experiments."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def retrieval_metrics(ranks: Sequence[int]) -> Dict[str, float]:
    arr = np.asarray(ranks, dtype=np.int32)
    valid = arr >= 0
    safe_ranks = np.where(valid, arr, 10**9)
    rr = np.zeros_like(arr, dtype=np.float32)
    rr[valid] = 1.0 / (arr[valid] + 1)
    return {
        "R@1": float((safe_ranks < 1).mean()) if arr.size else 0.0,
        "R@5": float((safe_ranks < 5).mean()) if arr.size else 0.0,
        "R@10": float((safe_ranks < 10).mean()) if arr.size else 0.0,
        "MRR": float(rr.mean()) if arr.size else 0.0,
        "MeanR": float(np.where(valid, arr + 1, len(arr)).mean()) if arr.size else 0.0,
        "MedR": float(np.median(np.where(valid, arr + 1, len(arr)))) if arr.size else 0.0,
    }


def summarize_results(
    method: str,
    ranks: Sequence[int],
    gt_filtered: Sequence[bool],
    costs: Sequence[float],
    conformal_sizes: Sequence[int] | None = None,
) -> Dict[str, float | str]:
    out: Dict[str, float | str] = {"method": method}
    out.update(retrieval_metrics(ranks))
    gf = np.asarray(gt_filtered, dtype=bool)
    co = np.asarray(costs, dtype=np.float32)
    out["GT_filtered_rate"] = float(gf.mean()) if gf.size else 0.0
    out["avg_cost"] = float(co.mean()) if co.size else 0.0
    if conformal_sizes is not None:
        cs = np.asarray(conformal_sizes, dtype=np.float32)
        out["avg_conformal_set_size"] = float(cs.mean()) if cs.size else 0.0
    return out


def submodularity_violation_rate(marginal_values: np.ndarray, subset_masks: np.ndarray) -> float:
    """Empirical violation rate for diminishing returns labels."""
    marg = np.asarray(marginal_values, dtype=np.float32)
    masks = np.asarray(subset_masks, dtype=bool)
    n_q, n_s, n_e = marg.shape
    total = 0
    violations = 0
    for small_idx in range(n_s):
        small = masks[small_idx]
        for large_idx in range(n_s):
            large = masks[large_idx]
            if not np.all(small <= large) or np.array_equal(small, large):
                continue
            for e in range(n_e):
                if large[e]:
                    continue
                small_vals = marg[:, small_idx, e]
                large_vals = marg[:, large_idx, e]
                ok = np.isfinite(small_vals) & np.isfinite(large_vals)
                total += int(ok.sum())
                violations += int((large_vals[ok] > small_vals[ok] + 1e-6).sum())
    return 0.0 if total == 0 else float(violations / total)
