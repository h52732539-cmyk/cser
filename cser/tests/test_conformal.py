from __future__ import annotations

import unittest

import numpy as np

from cser.conformal import MondrianConformalCalibrator, SplitConformalCalibrator


class TestConformal(unittest.TestCase):
    def test_split_coverage_on_calibration_like_data(self):
        n, m = 40, 20
        scores = np.zeros((n, m), dtype=np.float32)
        gt = np.arange(n) % m
        for i, g in enumerate(gt):
            scores[i] = np.linspace(0.0, 0.2, m)
            scores[i, g] = 1.0
        cal = SplitConformalCalibrator(alpha=0.1).fit(scores, gt)
        self.assertGreaterEqual(cal.coverage(scores, gt), 0.9)
        self.assertLess(cal.threshold, 0.2)

    def test_mondrian_fallback_for_small_bins(self):
        n, m = 12, 10
        rng = np.random.default_rng(0)
        scores = rng.random((n, m), dtype=np.float32)
        gt = np.argmax(scores, axis=1)
        cal = MondrianConformalCalibrator(alpha=0.1, n_bins=3, min_bin_size=30)
        cal.fit(scores, gt)
        self.assertEqual(len(cal.thresholds), 3)
        self.assertTrue(all(v == cal.global_calibrator.threshold for v in cal.thresholds.values()))


if __name__ == "__main__":
    unittest.main()
