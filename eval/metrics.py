"""Evaluation metrics for C-QIN and baselines.

All metrics operate on arrays of RouteResult-like dicts or raw arrays.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


# ----------------------------------------------------------------------
#  Retrieval metrics
# ----------------------------------------------------------------------

def recall_at_k(ranks: np.ndarray, k: int) -> float:
    if len(ranks) == 0:
        return 0.0
    return float(((ranks >= 0) & (ranks < k)).mean())


def mean_rank(ranks: np.ndarray) -> float:
    valid = ranks[ranks >= 0]
    return float(valid.mean() + 1) if len(valid) else float("inf")


def median_rank(ranks: np.ndarray) -> float:
    valid = ranks[ranks >= 0]
    return float(np.median(valid) + 1) if len(valid) else float("inf")


def mrr(ranks: np.ndarray) -> float:
    r = np.asarray(ranks)
    if len(r) == 0:
        return 0.0
    reciprocal_ranks = np.zeros(len(r), dtype=np.float64)
    valid = r >= 0
    reciprocal_ranks[valid] = 1.0 / (r[valid] + 1)
    return float(reciprocal_ranks.mean())


def retrieval_metrics(ranks: np.ndarray) -> Dict[str, float]:
    r = np.asarray(ranks, dtype=np.int32)
    return {
        "R@1":   recall_at_k(r, 1),
        "R@5":   recall_at_k(r, 5),
        "R@10":  recall_at_k(r, 10),
        "MeanR": mean_rank(r),
        "MedR":  median_rank(r),
        "MRR":   mrr(r),
    }


# ----------------------------------------------------------------------
#  Safety metrics
# ----------------------------------------------------------------------

def gt_filtered_rate(gt_filtered: np.ndarray) -> float:
    return float(np.asarray(gt_filtered, dtype=bool).mean())


def hard_filter_activation_rate(route_has_hard: np.ndarray) -> float:
    return float(np.asarray(route_has_hard, dtype=bool).mean())


def fallback_rate(used_fallback: np.ndarray) -> float:
    return float(np.asarray(used_fallback, dtype=bool).mean())


def safety_metrics(gt_filtered: np.ndarray,
                    route_has_hard: np.ndarray,
                    used_fallback: np.ndarray) -> Dict[str, float]:
    return {
        "GT_filtered_rate":          gt_filtered_rate(gt_filtered),
        "hard_filter_activation":    hard_filter_activation_rate(route_has_hard),
        "fallback_rate":             fallback_rate(used_fallback),
    }


# ----------------------------------------------------------------------
#  Efficiency metrics
# ----------------------------------------------------------------------

def cost_metrics(costs: np.ndarray,
                  latencies_ms: np.ndarray) -> Dict[str, float]:
    return {
        "avg_cost_proxy":  float(np.mean(costs)),
        "avg_ms_per_query": float(np.mean(latencies_ms)),
    }


# ----------------------------------------------------------------------
#  Oracle gap
# ----------------------------------------------------------------------

def oracle_gap(method_ranks: np.ndarray,
                oracle_ranks: np.ndarray) -> Dict[str, float]:
    m_r1 = recall_at_k(method_ranks, 1)
    o_r1 = recall_at_k(oracle_ranks, 1)
    m_mrr = mrr(method_ranks)
    o_mrr = mrr(oracle_ranks)
    return {
        "oracle_R@1":      o_r1,
        "method_R@1":      m_r1,
        "gap_R@1":         o_r1 - m_r1,
        "oracle_MRR":      o_mrr,
        "method_MRR":      m_mrr,
        "gap_MRR":         o_mrr - m_mrr,
    }


# ----------------------------------------------------------------------
#  Combined report
# ----------------------------------------------------------------------

def full_report(ranks: np.ndarray,
                 gt_filtered: np.ndarray,
                 route_has_hard: np.ndarray,
                 used_fallback: np.ndarray,
                 costs: np.ndarray,
                 latencies_ms: np.ndarray,
                 oracle_ranks: np.ndarray = None,
                 method_name: str = "") -> Dict[str, float]:
    out = {"method": method_name}
    out.update(retrieval_metrics(ranks))
    out.update(safety_metrics(gt_filtered, route_has_hard, used_fallback))
    out.update(cost_metrics(costs, latencies_ms))
    if oracle_ranks is not None:
        out.update(oracle_gap(ranks, oracle_ranks))
    return out
