"""Tests for route_schema.py."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing.route_schema import RetrievalRoute, FALLBACK_ROUTE, VALID_AXES


class TestRetrievalRoute(unittest.TestCase):
    def test_valid_route(self):
        r = RetrievalRoute(
            route_id="test", hard_axes=("time",), soft_axes=("geo",),
            candidate_topm=500, rerank_mode="nnn_qamp", budget_tier="medium",
        )
        self.assertEqual(r.route_id, "test")
        self.assertTrue(r.has_hard_filter)
        self.assertTrue(r.has_soft_rerank)

    def test_invalid_hard_axis(self):
        with self.assertRaises(ValueError):
            RetrievalRoute(route_id="bad", hard_axes=("semantic",))

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalRoute(
                route_id="bad",
                hard_axes=("time",), soft_axes=("time",),
            )

    def test_invalid_topm(self):
        with self.assertRaises(ValueError):
            RetrievalRoute(route_id="bad", candidate_topm=42)

    def test_low_budget_no_dense(self):
        with self.assertRaises(ValueError):
            RetrievalRoute(
                route_id="bad", budget_tier="low",
                allow_dense_refinement=True,
            )

    def test_high_budget_allows_dense(self):
        r = RetrievalRoute(
            route_id="ok", budget_tier="high",
            allow_dense_refinement=True,
        )
        self.assertTrue(r.allow_dense_refinement)

    def test_from_dict_round_trip(self):
        r = RetrievalRoute(
            route_id="R06", hard_axes=("time", "geo"),
            candidate_topm=500, rerank_mode="col_softmax_post_filter",
            budget_tier="low",
        )
        d = r.to_dict()
        r2 = RetrievalRoute.from_dict(d)
        self.assertEqual(r.route_id, r2.route_id)
        self.assertEqual(r.hard_axes, r2.hard_axes)

    def test_fallback_is_valid(self):
        self.assertEqual(FALLBACK_ROUTE.hard_axes, ())
        self.assertEqual(FALLBACK_ROUTE.budget_tier, "low")
        self.assertFalse(FALLBACK_ROUTE.has_hard_filter)

    def test_cost_tier_value(self):
        r_low = RetrievalRoute(route_id="l", budget_tier="low")
        r_full = RetrievalRoute(route_id="f", budget_tier="full")
        self.assertLess(r_low.cost_tier_value, r_full.cost_tier_value)


if __name__ == "__main__":
    unittest.main()
