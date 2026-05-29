"""Baselines B0-B8 for AAAI comparison.

Each baseline is a callable: (query_emb, gt_vid, intent, executor, bank) → RouteResult.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import numpy as np

from .route_schema import RetrievalRoute, FALLBACK_ROUTE
from .route_bank import RouteBank
from .route_executor import RouteExecutor, RouteResult
from .route_bank_builder import compute_utility

import sys
from pathlib import Path
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
from core.query_parser import QueryIntent, QueryParser


# Type alias for a baseline function
BaselineFn = Callable[
    ["np.ndarray", str, "QueryIntent", "RouteExecutor", "RouteBank"],
    RouteResult
]


# B0: Semantic-only (Phase 2 baseline)
def b0_semantic_only(query_emb, gt_vid, intent, executor, bank):
    return executor.execute(FALLBACK_ROUTE, query_emb, gt_vid, intent)


# B1: Rule-based QueryParser + MetaFilter (Phase 3 baseline)
def b1_rule_parser(query_emb, gt_vid, intent, executor, bank):
    if not intent.has_constraint():
        return executor.execute(FALLBACK_ROUTE, query_emb, gt_vid, intent)
    # Pick route matching detected axes
    hard = []
    if intent.time_window is not None:
        hard.append("time")
    if intent.geo_categories:
        hard.append("geo")
    if intent.motion_classes:
        hard.append("motion")
    if intent.device_filter:
        hard.append("device")
    # Find best matching route in bank
    best_route = FALLBACK_ROUTE
    best_match = 0
    for r in bank:
        match = len(set(r.hard_axes) & set(hard))
        extra = len(set(r.hard_axes) - set(hard))
        if match > best_match and extra == 0:
            best_match = match
            best_route = r
    return executor.execute(best_route, query_emb, gt_vid, intent)


# B2: QPP-only router (margin-based)
def b2_qpp_only(query_emb, gt_vid, intent, executor, bank):
    res_sem = executor.execute(FALLBACK_ROUTE, query_emb, gt_vid, intent)
    hits_list = executor.index.search_batch(
        query_emb[np.newaxis], top_k=5, col_beta=0.0, topm_rerank=100,
    )[0]
    if len(hits_list) >= 2:
        margin = hits_list[0][1] - hits_list[1][1]
    else:
        margin = 0.0
    if margin > 0.08:
        return res_sem
    elif margin > 0.02:
        r = bank.get("R16_semantic_top1000_nnn_qamp") or FALLBACK_ROUTE
        return executor.execute(r, query_emb, gt_vid, intent)
    else:
        r = bank.get("R29_full_budget_all_soft_dense_refine") or FALLBACK_ROUTE
        return executor.execute(r, query_emb, gt_vid, intent)


# B3: Random route
class B3RandomRoute:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def __call__(self, query_emb, gt_vid, intent, executor, bank):
        route = self.rng.choice(bank.routes)
        return executor.execute(route, query_emb, gt_vid, intent)


# B4: Oracle route (argmax utility per query)
def b4_oracle_route(query_emb, gt_vid, intent, executor, bank):
    best_res = None
    best_util = -float("inf")
    for route in bank:
        res = executor.execute(route, query_emb, gt_vid, intent)
        u = compute_utility(res.rank, res.gt_filtered, res.cost_proxy)
        if u > best_util:
            best_util = u
            best_res = res
    return best_res


# B5: Always-hard-filter-all detected axes
def b5_always_hard_all(query_emb, gt_vid, intent, executor, bank):
    hard = []
    if intent.time_window is not None:
        hard.append("time")
    if intent.geo_categories:
        hard.append("geo")
    if intent.motion_classes:
        hard.append("motion")
    if intent.device_filter:
        hard.append("device")
    if not hard:
        return executor.execute(FALLBACK_ROUTE, query_emb, gt_vid, intent)
    # Find route with exactly these hard_axes
    for r in bank:
        if set(r.hard_axes) == set(hard) and r.rerank_mode == "col_softmax_post_filter":
            return executor.execute(r, query_emb, gt_vid, intent)
    # Fallback to the one with most matching axes
    best = FALLBACK_ROUTE
    best_n = 0
    for r in bank:
        n = len(set(r.hard_axes) & set(hard))
        if n > best_n and not (set(r.hard_axes) - set(hard)):
            best = r
            best_n = n
    return executor.execute(best, query_emb, gt_vid, intent)


# B6: C-QIN without calibration (argmax route_value, no safety gate)
def make_b6_uncalibrated(model, bank, device="cpu"):
    import torch
    model = model.eval().to(torch.device(device))

    def b6(query_emb, gt_vid, intent, executor, _bank):
        import torch as _t
        from .qin_model import extract_qin_features
        feat = extract_qin_features(
            "", query_emb,
            np.zeros(20, dtype=np.float32),
            intent, np.zeros(4, dtype=np.float32),
        )
        x = _t.from_numpy(feat).float().unsqueeze(0).to(device)
        with _t.no_grad():
            out = model(x)
        vals = out["route_values"][0].cpu().numpy()
        best_idx = int(np.argmax(vals))
        route = bank.routes[best_idx]
        return executor.execute(route, query_emb, gt_vid, intent)
    return b6


# B7: C-QIN with calibration (main method — delegates to CalibratedPlanner)
def make_b7_calibrated(planner):
    from .calibrated_planner import CalibratedPlanner
    from .qin_model import extract_qin_features

    def b7(query_emb, gt_vid, intent, executor, bank):
        feat = extract_qin_features(
            "", query_emb,
            np.zeros(20, dtype=np.float32),
            intent, np.zeros(4, dtype=np.float32),
        )
        decision, result = planner.plan_and_execute(
            feat, query_emb, gt_vid, intent, executor,
        )
        return result
    return b7


# B8: Cascade baseline
def b8_cascade(query_emb, gt_vid, intent, executor, bank):
    # Stage 1: semantic only, low budget
    r1 = bank.get("R01_semantic_only_top300") or FALLBACK_ROUTE
    res1 = executor.execute(r1, query_emb, gt_vid, intent)
    if res1.rank == 0:
        return res1
    # Stage 2: if intent has constraints, try hard filter
    if intent.has_constraint():
        res2 = b1_rule_parser(query_emb, gt_vid, intent, executor, bank)
        if res2.rank >= 0 and res2.rank < res1.rank:
            return res2
    # Stage 3: high budget
    r3 = bank.get("R16_semantic_top1000_nnn_qamp") or FALLBACK_ROUTE
    res3 = executor.execute(r3, query_emb, gt_vid, intent)
    best = min([res1, res2 if intent.has_constraint() else res1, res3],
                key=lambda r: r.rank if r.rank >= 0 else 9999)
    return best


# Convenience: get all baselines as a dict
def get_all_baselines(model=None, planner=None, seed=42) -> Dict[str, BaselineFn]:
    out = {
        "B0_semantic_only": b0_semantic_only,
        "B1_rule_parser": b1_rule_parser,
        "B2_qpp_only": b2_qpp_only,
        "B3_random": B3RandomRoute(seed),
        "B4_oracle": b4_oracle_route,
        "B5_always_hard_all": b5_always_hard_all,
        "B8_cascade": b8_cascade,
    }
    if model is not None:
        out["B6_cqin_uncalibrated"] = make_b6_uncalibrated(model, RouteBank.from_yaml())
    if planner is not None:
        out["B7_cqin_calibrated"] = make_b7_calibrated(planner)
    return out
