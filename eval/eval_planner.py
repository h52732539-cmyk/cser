"""Evaluate any baseline/method on a dataset and produce a results row."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np

from routing.route_bank import RouteBank
from routing.route_executor import RouteExecutor, RouteResult
from routing.baselines import BaselineFn
from eval.metrics import full_report

import sys
from pathlib import Path
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
from core.query_parser import QueryIntent


def evaluate_method(
    method_name: str,
    method_fn: BaselineFn,
    query_embs: np.ndarray,
    gt_video_ids: List[str],
    intents: List[QueryIntent],
    executor: RouteExecutor,
    bank: RouteBank,
    oracle_ranks: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict:
    """Run one method on all queries and return a full metrics dict."""
    N = len(gt_video_ids)
    ranks = np.full(N, -1, dtype=np.int32)
    gt_filtered = np.zeros(N, dtype=bool)
    route_has_hard = np.zeros(N, dtype=bool)
    used_fallback = np.zeros(N, dtype=bool)
    costs = np.zeros(N, dtype=np.float32)
    latencies = np.zeros(N, dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(N):
        try:
            res = method_fn(
                query_embs[i], gt_video_ids[i], intents[i],
                executor, bank,
            )
            ranks[i] = res.rank
            gt_filtered[i] = res.gt_filtered
            route_has_hard[i] = len(res.route_id) > 0 and "hard" in res.route_id
            costs[i] = res.cost_proxy
            latencies[i] = res.latency_ms
        except Exception as e:
            if verbose:
                print(f"  [warn] query {i}: {e}")
            ranks[i] = -1
            gt_filtered[i] = True

        if verbose and (i + 1) % 200 == 0:
            r1 = float(((ranks[:i+1] >= 0) & (ranks[:i+1] < 1)).mean())
            print(f"  [{i+1}/{N}] R@1={r1:.3f}")

    dt = (time.perf_counter() - t0) * 1000.0
    if verbose:
        print(f"  {method_name}: total={dt:.0f}ms  ({dt/N:.2f}ms/q)")

    # Check which queries actually used routes with hard filter
    # (override the heuristic above with actual route info if available)

    report = full_report(
        ranks=ranks,
        gt_filtered=gt_filtered,
        route_has_hard=route_has_hard,
        used_fallback=used_fallback,
        costs=costs,
        latencies_ms=latencies,
        oracle_ranks=oracle_ranks,
        method_name=method_name,
    )
    report["total_ms"] = dt
    return report


def evaluate_all_methods(
    methods: Dict[str, BaselineFn],
    query_embs: np.ndarray,
    gt_video_ids: List[str],
    intents: List[QueryIntent],
    executor: RouteExecutor,
    bank: RouteBank,
    oracle_ranks: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Evaluate multiple methods and return a list of result dicts."""
    results = []
    for name, fn in methods.items():
        if verbose:
            print(f"\n=== {name} ===")
        r = evaluate_method(
            name, fn, query_embs, gt_video_ids, intents,
            executor, bank, oracle_ranks=oracle_ranks, verbose=verbose,
        )
        results.append(r)
    return results
