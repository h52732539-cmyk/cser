"""Tests for route_bank.py."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing.route_bank import RouteBank
from routing.route_schema import FALLBACK_ROUTE


class TestRouteBank(unittest.TestCase):
    def setUp(self):
        self.bank = RouteBank.from_yaml()

    def test_bank_size(self):
        self.assertGreaterEqual(len(self.bank), 25)
        self.assertLessEqual(len(self.bank), 40)

    def test_all_ids_unique(self):
        ids = self.bank.ids
        self.assertEqual(len(ids), len(set(ids)))

    def test_has_fallback(self):
        fb = self.bank.fallback
        self.assertEqual(fb.route_id, FALLBACK_ROUTE.route_id)
        self.assertFalse(fb.has_hard_filter)

    def test_lookup_by_id(self):
        r = self.bank["R00_semantic_only_top500"]
        self.assertEqual(r.route_id, "R00_semantic_only_top500")

    def test_missing_id_raises(self):
        with self.assertRaises(KeyError):
            _ = self.bank["NONEXISTENT"]

    def test_routes_with_hard_axis(self):
        time_routes = self.bank.routes_with_hard_axis("time")
        self.assertGreater(len(time_routes), 0)
        for r in time_routes:
            self.assertIn("time", r.hard_axes)

    def test_index_of(self):
        idx = self.bank.index_of("R00_semantic_only_top500")
        self.assertGreaterEqual(idx, 0)

    def test_summary(self):
        s = self.bank.summary()
        self.assertIn("n_routes", s)
        self.assertIn("budget_dist", s)
        self.assertGreater(s["n_routes"], 0)

    def test_all_routes_valid(self):
        for r in self.bank:
            r.validate()


if __name__ == "__main__":
    unittest.main()
