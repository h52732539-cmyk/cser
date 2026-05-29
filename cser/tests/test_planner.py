from __future__ import annotations

import unittest

import numpy as np

from cser.expert_store import ExpertOutputStore
from cser.planner import GreedyBudgetedSelector
from cser.subset_executor import CSERSubsetExecutor


class DummyGate:
    def __init__(self, mask):
        self.mask = mask

    def predict(self, scores, difficulty=None):
        return self.mask


def dummy_values(features, selected_mask):
    return np.asarray([0.0, 0.2, 0.8, 0.1, 0.6], dtype=np.float32)


class TestPlanner(unittest.TestCase):
    def test_budget_compliance(self):
        store = ExpertOutputStore.synthetic(n_videos=16, seed=1)
        executor = CSERSubsetExecutor(store)
        selector = GreedyBudgetedSelector(dummy_values, tau_stop=0.0)
        decision, _ = selector.plan(
            np.zeros(536, dtype=np.float32),
            store.clip_video_embs[0],
            budget=3.0,
            executor=executor,
        )
        self.assertLessEqual(decision.budget_used, 3.0)
        self.assertIn("clip_semantic", decision.selected_experts)

    def test_protected_video_not_filtered(self):
        store = ExpertOutputStore.synthetic(n_videos=12, seed=2)
        executor = CSERSubsetExecutor(store)
        gt = store.video_ids[0]
        protected = np.zeros(store.size, dtype=bool)
        protected[0] = True
        selector = GreedyBudgetedSelector(dummy_values, conformal_gate=DummyGate(protected), tau_stop=0.0)
        ctx = {"scene_label": "not_a_real_scene", "requires_scene_filter": True}
        decision, result = selector.plan_and_execute(
            np.zeros(536, dtype=np.float32),
            store.clip_video_embs[0],
            gt,
            budget=3.0,
            executor=executor,
            query_context=ctx,
        )
        self.assertFalse(result.gt_filtered)
        self.assertGreaterEqual(decision.conformal_set_size, 1)


if __name__ == "__main__":
    unittest.main()
