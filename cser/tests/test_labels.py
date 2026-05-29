from __future__ import annotations

import unittest

import numpy as np

from cser.expert_store import ExpertOutputStore
from cser.labels import build_oracle_labels, enumerate_valid_subsets
from cser.subset_executor import CSERSubsetExecutor


class TestLabels(unittest.TestCase):
    def test_enumerates_mandatory_subsets(self):
        subsets = enumerate_valid_subsets()
        self.assertEqual(subsets.shape[0], 16)
        self.assertTrue(np.all(subsets[:, 0]))

    def test_build_labels_shapes(self):
        store = ExpertOutputStore.synthetic(n_videos=20, seed=3)
        executor = CSERSubsetExecutor(store)
        query_embs = store.clip_video_embs[:4]
        gt = store.video_ids[:4]
        ctx = [store.query_context_for_gt(g, np.random.default_rng(i)) for i, g in enumerate(gt)]
        labels = build_oracle_labels(executor, query_embs, gt, ctx)
        self.assertEqual(labels.qualities.shape, (4, 16))
        self.assertEqual(labels.marginal_values.shape, (4, 16, 5))


if __name__ == "__main__":
    unittest.main()
