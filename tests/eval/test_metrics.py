"""Tests for eval/metrics.py."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from eval.metrics import (
    recall_at_k, mean_rank, mrr, retrieval_metrics,
    gt_filtered_rate, safety_metrics, full_report,
)


class TestRetrievalMetrics(unittest.TestCase):
    def test_perfect_recall(self):
        ranks = np.array([0, 0, 0, 0])
        self.assertAlmostEqual(recall_at_k(ranks, 1), 1.0)
        self.assertAlmostEqual(recall_at_k(ranks, 5), 1.0)

    def test_zero_recall(self):
        ranks = np.array([100, 200, 300])
        self.assertAlmostEqual(recall_at_k(ranks, 1), 0.0)

    def test_partial_recall(self):
        ranks = np.array([0, 3, 8, 50])
        self.assertAlmostEqual(recall_at_k(ranks, 5), 0.5)

    def test_mean_rank(self):
        ranks = np.array([0, 4, 9])  # 0-based → +1 for display
        self.assertAlmostEqual(mean_rank(ranks), (1 + 5 + 10) / 3)

    def test_mrr(self):
        ranks = np.array([0, 1, 4])  # MRR = (1 + 1/2 + 1/5) / 3
        expected = (1.0 + 0.5 + 0.2) / 3
        self.assertAlmostEqual(mrr(ranks), expected, places=4)

    def test_filtered_ranks(self):
        ranks = np.array([0, -1, 2])  # -1 = filtered out
        self.assertAlmostEqual(recall_at_k(ranks, 1), 1 / 3)


class TestSafetyMetrics(unittest.TestCase):
    def test_gt_filtered_rate(self):
        gt = np.array([True, False, False, True])
        self.assertAlmostEqual(gt_filtered_rate(gt), 0.5)

    def test_safety_metrics_combined(self):
        m = safety_metrics(
            gt_filtered=np.array([True, False, False]),
            route_has_hard=np.array([True, True, False]),
            used_fallback=np.array([False, False, True]),
        )
        self.assertAlmostEqual(m["GT_filtered_rate"], 1 / 3)
        self.assertAlmostEqual(m["hard_filter_activation"], 2 / 3)
        self.assertAlmostEqual(m["fallback_rate"], 1 / 3)


class TestFullReport(unittest.TestCase):
    def test_has_all_keys(self):
        r = full_report(
            ranks=np.array([0, 1, 5, -1]),
            gt_filtered=np.array([False, False, False, True]),
            route_has_hard=np.array([True, True, False, True]),
            used_fallback=np.array([False, False, False, True]),
            costs=np.array([1.0, 2.0, 1.0, 4.0]),
            latencies_ms=np.array([5.0, 10.0, 3.0, 20.0]),
            method_name="test",
        )
        self.assertIn("R@1", r)
        self.assertIn("MRR", r)
        self.assertIn("GT_filtered_rate", r)
        self.assertIn("avg_ms_per_query", r)
        self.assertEqual(r["method"], "test")


if __name__ == "__main__":
    unittest.main()
