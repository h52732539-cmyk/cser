"""Tests for route_executor.py."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from routing.route_schema import RetrievalRoute, FALLBACK_ROUTE
from routing.route_executor import RouteExecutor, RouteResult
from routing.route_bank_builder import compute_utility
from core.offline_index import OfflineIndex, VideoIndexEntry, build_protos
from core.metadata import VideoMetadata
from core.query_parser import QueryIntent
from core.meta_filter import MetaFilter

from tests.routing import _mk_test_index


class TestRouteExecutor(unittest.TestCase):
    def setUp(self):
        self.index, self.gt_map = _mk_test_index()
        self.executor = RouteExecutor(self.index)

    def test_semantic_only_finds_gt(self):
        q = np.zeros(8, dtype=np.float32); q[0] = 1.0
        res = self.executor.execute(FALLBACK_ROUTE, q, "v0")
        self.assertEqual(res.rank, 0)
        self.assertFalse(res.gt_filtered)
        self.assertEqual(res.recall_at[1], 1)

    def test_hard_filter_can_eliminate_gt(self):
        route = RetrievalRoute(
            route_id="test_hard", hard_axes=("geo",),
            candidate_topm=500, rerank_mode="col_softmax_post_filter",
            budget_tier="low",
        )
        # v0 is "coast", query for "mountain" → v0 should be filtered
        q = np.zeros(8, dtype=np.float32); q[0] = 1.0
        intent = QueryIntent(semantic_text="", geo_categories=["mountain"])
        res = self.executor.execute(route, q, "v0", intent)
        self.assertTrue(res.gt_filtered)
        self.assertEqual(res.rank, -1)

    def test_survival_labels(self):
        intent = QueryIntent(
            semantic_text="",
            geo_categories=["coast"],
            motion_classes=["running"],
        )
        labels = self.executor.survival_labels("v0", intent)
        self.assertEqual(labels["geo"], 1)  # v0 is coast
        self.assertEqual(labels["motion"], 1)  # v0 is running

    def test_cost_proxy(self):
        r_low = RetrievalRoute(route_id="l", budget_tier="low")
        r_high = RetrievalRoute(route_id="h", budget_tier="high")
        q = np.zeros(8, dtype=np.float32); q[0] = 1.0
        res_l = self.executor.execute(r_low, q, "v0")
        res_h = self.executor.execute(r_high, q, "v0")
        self.assertLess(res_l.cost_proxy, res_h.cost_proxy)


class TestUtility(unittest.TestCase):
    def test_hit1(self):
        u = compute_utility(rank=0, gt_filtered=False, cost=1.0)
        self.assertGreater(u, 0)

    def test_filtered_penalty(self):
        u = compute_utility(rank=-1, gt_filtered=True, cost=1.0)
        self.assertLess(u, 0)

    def test_higher_rank_lower_utility(self):
        u1 = compute_utility(rank=0, gt_filtered=False, cost=1.0)
        u10 = compute_utility(rank=9, gt_filtered=False, cost=1.0)
        self.assertGreater(u1, u10)


if __name__ == "__main__":
    unittest.main()
