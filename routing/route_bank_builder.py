"""Route bank builder — generate counterfactual labels for C-QIN training.

For each (query, route) pair, executes the route and records:
  - rank of GT video
  - gt_filtered (bool)
  - cost_proxy
  - utility score
  - per-axis survival labels

Output: a numpy archive (.npz) containing all labels needed for training.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .route_schema import RetrievalRoute
from .route_bank import RouteBank
from .route_executor import RouteExecutor, RouteResult

import sys
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from core.offline_index import OfflineIndex
from core.meta_filter import MetaFilter
from core.query_parser import QueryIntent


# Utility function (matches GPT spec)
def compute_utility(rank: int, gt_filtered: bool, cost: float,
                     gain_weight: float = 1.0,
                     hit1_weight: float = 0.5,
                     hit5_weight: float = 0.1,
                     cost_weight: float = 0.05,
                     filter_penalty: float = 2.0) -> float:
    if gt_filtered or rank < 0:
        return -filter_penalty
    gain = 1.0 / (rank + 1)  # MRR-style
    hit1 = int(rank == 0)
    hit5 = int(rank < 5)
    return (gain_weight * gain + hit1_weight * hit1 +
            hit5_weight * hit5 - cost_weight * cost)


@dataclass
class RouteBankLabels:
    """All counterfactual labels for a dataset."""
    n_queries: int
    n_routes: int
    route_ids: List[str]

    # (n_queries, n_routes) matrices
    ranks: np.ndarray           # int, 0-based (-1 if filtered)
    gt_filtered: np.ndarray     # bool
    utilities: np.ndarray       # float
    costs: np.ndarray           # float

    # (n_queries,) oracle
    oracle_route_idx: np.ndarray  # int, index into route_ids
    oracle_utility: np.ndarray    # float

    # (n_queries, 4) survival labels per axis [time, geo, motion, device]
    survival_labels: np.ndarray   # bool

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ranks=self.ranks,
            gt_filtered=self.gt_filtered,
            utilities=self.utilities,
            costs=self.costs,
            oracle_route_idx=self.oracle_route_idx,
            oracle_utility=self.oracle_utility,
            survival_labels=self.survival_labels,
            route_ids=np.array(self.route_ids),
            n_queries=self.n_queries,
            n_routes=self.n_routes,
        )
        # companion json
        summary = {
            "n_queries": self.n_queries,
            "n_routes": self.n_routes,
            "oracle_mean_utility": float(self.oracle_utility.mean()),
            "oracle_R@1": float((self.ranks[np.arange(self.n_queries),
                                             self.oracle_route_idx] == 0).mean()),
            "gt_filtered_rate_mean": float(self.gt_filtered.mean()),
        }
        Path(path + ".json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str) -> "RouteBankLabels":
        d = np.load(path, allow_pickle=True)
        return cls(
            n_queries=int(d["n_queries"]),
            n_routes=int(d["n_routes"]),
            route_ids=list(d["route_ids"]),
            ranks=d["ranks"],
            gt_filtered=d["gt_filtered"],
            utilities=d["utilities"],
            costs=d["costs"],
            oracle_route_idx=d["oracle_route_idx"],
            oracle_utility=d["oracle_utility"],
            survival_labels=d["survival_labels"],
        )


# ----------------------------------------------------------------------

def build_route_bank_labels(
    index: OfflineIndex,
    bank: RouteBank,
    query_embs: np.ndarray,
    gt_video_ids: List[str],
    intents: List[QueryIntent],
    meta_filter: Optional[MetaFilter] = None,
    verbose: bool = True,
) -> RouteBankLabels:
    """Execute all routes for all queries → produce counterfactual labels."""
    Nq = len(gt_video_ids)
    Nr = len(bank)
    route_ids = bank.ids
    executor = RouteExecutor(index, meta_filter=meta_filter)

    ranks = np.full((Nq, Nr), -1, dtype=np.int32)
    gt_filt = np.zeros((Nq, Nr), dtype=bool)
    utilities = np.zeros((Nq, Nr), dtype=np.float32)
    costs = np.zeros((Nq, Nr), dtype=np.float32)
    survival = np.zeros((Nq, 4), dtype=bool)

    t0 = time.perf_counter()
    for qi in range(Nq):
        q_emb = query_embs[qi]
        gt_vid = gt_video_ids[qi]
        intent = intents[qi]

        # Survival labels (per-axis, computed once per query)
        slabels = executor.survival_labels(gt_vid, intent)
        survival[qi] = [
            slabels.get("time", 1),
            slabels.get("geo", 1),
            slabels.get("motion", 1),
            slabels.get("device", 1),
        ]

        # Execute each route
        for ri, route in enumerate(bank):
            res = executor.execute(route, q_emb, gt_vid, intent)
            ranks[qi, ri] = res.rank
            gt_filt[qi, ri] = res.gt_filtered
            costs[qi, ri] = res.cost_proxy
            utilities[qi, ri] = compute_utility(
                res.rank, res.gt_filtered, res.cost_proxy,
            )

        if verbose and (qi + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (qi + 1) * (Nq - qi - 1)
            print(f"  [{qi+1}/{Nq}] elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    # Oracle route per query
    oracle_idx = utilities.argmax(axis=1)
    oracle_util = utilities[np.arange(Nq), oracle_idx]

    return RouteBankLabels(
        n_queries=Nq,
        n_routes=Nr,
        route_ids=route_ids,
        ranks=ranks,
        gt_filtered=gt_filt,
        utilities=utilities,
        costs=costs,
        oracle_route_idx=oracle_idx,
        oracle_utility=oracle_util,
        survival_labels=survival,
    )
