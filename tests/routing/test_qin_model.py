"""Tests for qin_model.py and train_qin.py."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from routing.qin_model import CalibratedQIN, extract_qin_features
from routing.train_qin import TrainConfig, train_cqin, _normalize_utilities
from routing.route_bank_builder import RouteBankLabels


class TestCalibratedQIN(unittest.TestCase):
    def test_output_shapes(self):
        model = CalibratedQIN(input_dim=531, num_routes=30, num_safety_axes=4)
        x = torch.randn(8, 531)
        out = model(x)
        self.assertEqual(out["route_values"].shape, (8, 30))
        self.assertEqual(out["safety_logits"].shape, (8, 4))
        self.assertEqual(out["safety_probs"].shape, (8, 4))

    def test_param_count_under_100k(self):
        model = CalibratedQIN(input_dim=531, num_routes=30)
        self.assertLess(model.param_count(), 100_000)

    def test_safety_probs_in_01(self):
        model = CalibratedQIN(input_dim=531, num_routes=30)
        x = torch.randn(4, 531)
        out = model(x)
        self.assertTrue((out["safety_probs"] >= 0).all())
        self.assertTrue((out["safety_probs"] <= 1).all())

    def test_different_input_dims(self):
        for dim in (100, 531, 1024):
            model = CalibratedQIN(input_dim=dim, num_routes=10)
            x = torch.randn(2, dim)
            out = model(x)
            self.assertEqual(out["route_values"].shape[1], 10)


class TestFeatureExtraction(unittest.TestCase):
    def test_output_length(self):
        from core.query_parser import QueryIntent
        feat = extract_qin_features(
            query_text="test query",
            query_emb=np.random.randn(512).astype(np.float32),
            sem_scores_top20=np.random.randn(20).astype(np.float32),
            intent=QueryIntent(semantic_text="test"),
            meta_availability=np.array([0.8, 0.3, 0.5, 0.1]),
            budget_tier="low",
        )
        self.assertEqual(len(feat), 531)  # 512+6+5+4+4

    def test_budget_one_hot(self):
        from core.query_parser import QueryIntent
        for tier, idx in [("low", 0), ("medium", 1), ("high", 2), ("full", 3)]:
            feat = extract_qin_features(
                "q", np.zeros(512), np.zeros(20),
                QueryIntent(semantic_text=""), np.zeros(4),
                budget_tier=tier,
            )
            budget = feat[-4:]
            self.assertEqual(budget[idx], 1.0)
            self.assertEqual(sum(budget), 1.0)


class TestTrainSmoke(unittest.TestCase):
    def test_tiny_train(self):
        Nq, Nr = 50, 5
        features = np.random.randn(Nq, 531).astype(np.float32)
        labels = RouteBankLabels(
            n_queries=Nq, n_routes=Nr,
            route_ids=[f"R{i}" for i in range(Nr)],
            ranks=np.random.randint(0, 100, (Nq, Nr)),
            gt_filtered=np.random.choice([True, False], (Nq, Nr)),
            utilities=np.random.randn(Nq, Nr).astype(np.float32),
            costs=np.ones((Nq, Nr), dtype=np.float32),
            oracle_route_idx=np.random.randint(0, Nr, Nq),
            oracle_utility=np.random.randn(Nq).astype(np.float32),
            survival_labels=np.random.choice(
                [True, False], (Nq, 4)).astype(bool),
        )
        cfg = TrainConfig(epochs=3, batch_size=16, patience=5)
        model, history = train_cqin(features, labels, cfg, verbose=False)
        self.assertIsInstance(model, CalibratedQIN)
        self.assertGreater(len(history["train_loss"]), 0)
        self.assertLess(model.param_count(), 100_000)


class TestNormalizeUtilities(unittest.TestCase):
    def test_range_01(self):
        u = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        n = _normalize_utilities(u)
        self.assertAlmostEqual(n[0, 0], 0.0)
        self.assertAlmostEqual(n[0, 2], 1.0)
        # constant row → all 0
        self.assertTrue(np.allclose(n[1], 0.0))


if __name__ == "__main__":
    unittest.main()
