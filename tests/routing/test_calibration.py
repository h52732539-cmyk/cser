"""Tests for calibrate_safety.py and calibrated_planner.py."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from routing.calibrate_safety import (
    calibrate_one_axis, calibrate_all_axes, CalibrationResult,
    save_calibration, load_calibration, clopper_pearson_upper_fallback,
)
from routing.calibrated_planner import CalibratedPlanner, PlannerDecision
from routing.qin_model import CalibratedQIN
from routing.route_bank import RouteBank
from routing.route_schema import FALLBACK_ROUTE


class TestCalibration(unittest.TestCase):
    def test_perfect_safety(self):
        # All safety scores high, no failures → tau should be low
        scores = np.ones(100, dtype=np.float32) * 0.9
        failures = np.zeros(100, dtype=np.float32)
        r = calibrate_one_axis(scores, failures, delta=0.05, min_accept=10)
        self.assertTrue(r.enabled)
        self.assertLessEqual(r.tau, 0.91)

    def test_all_failures(self):
        # All failures → no threshold can satisfy δ=0.05
        scores = np.linspace(0.1, 0.9, 50).astype(np.float32)
        failures = np.ones(50, dtype=np.float32)
        r = calibrate_one_axis(scores, failures, delta=0.05, min_accept=10)
        self.assertFalse(r.enabled)

    def test_insufficient_samples(self):
        scores = np.array([0.5, 0.6], dtype=np.float32)
        failures = np.array([0, 0], dtype=np.float32)
        r = calibrate_one_axis(scores, failures, min_accept=30)
        self.assertFalse(r.enabled)

    def test_calibrate_all_axes(self):
        N = 100
        safety_probs = np.random.rand(N, 4).astype(np.float32)
        survival = np.random.choice([0, 1], (N, 4), p=[0.1, 0.9]).astype(np.float32)
        results = calibrate_all_axes(safety_probs, survival,
                                      delta=0.10, min_accept=10)
        self.assertEqual(len(results), 4)
        for axis in ("time", "geo", "motion", "device"):
            self.assertIn(axis, results)
            self.assertEqual(results[axis].axis, axis)

    def test_save_load_roundtrip(self):
        import tempfile
        results = {
            "time": CalibrationResult("time", 0.7, True, 80, 100, 0.02, 0.04),
            "geo":  CalibrationResult("geo", 1.0, False, 0, 100, 0.0, 1.0),
            "motion": CalibrationResult("motion", 0.5, True, 90, 100, 0.01, 0.03),
            "device": CalibrationResult("device", 1.0, False, 0, 100, 0.0, 1.0),
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cal.json"
            save_calibration(results, str(p))
            loaded = load_calibration(str(p))
            self.assertEqual(loaded["time"].tau, 0.7)
            self.assertTrue(loaded["time"].enabled)
            self.assertFalse(loaded["geo"].enabled)

    def test_wilson_fallback(self):
        ucb = clopper_pearson_upper_fallback(2, 100, alpha=0.05)
        self.assertGreater(ucb, 0.02)
        self.assertLess(ucb, 0.10)


class TestCalibratedPlanner(unittest.TestCase):
    def setUp(self):
        self.bank = RouteBank.from_yaml()
        self.model = CalibratedQIN(input_dim=531, num_routes=len(self.bank))
        # Enable time axis, disable others
        self.calibration = {
            "time": CalibrationResult("time", 0.3, True, 80, 100, 0.02, 0.04),
            "geo":  CalibrationResult("geo", 1.0, False, 0, 100, 0.0, 1.0),
            "motion": CalibrationResult("motion", 1.0, False, 0, 100, 0.0, 1.0),
            "device": CalibrationResult("device", 1.0, False, 0, 100, 0.0, 1.0),
        }

    def test_planner_returns_decision(self):
        planner = CalibratedPlanner(self.model, self.bank, self.calibration)
        feat = np.random.randn(531).astype(np.float32)
        decision = planner.plan(feat)
        self.assertIsInstance(decision, PlannerDecision)
        self.assertIsNotNone(decision.selected_route)
        self.assertGreater(decision.n_total_routes, 0)

    def test_fallback_when_no_safe(self):
        # All axes disabled except time (tau very high)
        cal = {a: CalibrationResult(a, 999.0, True, 100, 100, 0, 0)
                for a in ("time", "geo", "motion", "device")}
        planner = CalibratedPlanner(self.model, self.bank, cal)
        feat = np.random.randn(531).astype(np.float32)
        decision = planner.plan(feat)
        # Only routes WITHOUT hard_axes should be safe
        # Fallback (semantic-only) should always be safe
        self.assertFalse(decision.selected_route.has_hard_filter)

    def test_semantic_only_always_safe(self):
        # Even with impossible thresholds, fallback is always available
        cal = {a: CalibrationResult(a, 999.0, True, 100, 100, 0, 0)
                for a in ("time", "geo", "motion", "device")}
        planner = CalibratedPlanner(self.model, self.bank, cal)
        feat = np.random.randn(531).astype(np.float32)
        decision = planner.plan(feat)
        # Semantic-only has no hard_axes → always passes safety gate
        self.assertGreaterEqual(decision.n_safe_routes, 1)


if __name__ == "__main__":
    unittest.main()
