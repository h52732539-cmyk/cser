"""Unit tests for v2 components (black-box-model-preserving).

Run:   python -m unittest tests.test_v2 -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.frame_identity import FrameIdentity, byte_hash, phash, hamming
from core.cross_task_cache import CrossTaskCache
from core.adaptive_sampler import (
    UniformSampler, HybridSampler, QFrameSampler, QFrameConfig,
)
from core.offline_index import (
    OfflineIndex, OfflineIndexBuilder, build_protos, VideoIndexEntry,
)
from core.query_planner import (
    QueryPlanner, QueryPlannerConfig, QueryDifficulty,
)


class TestFrameIdentity(unittest.TestCase):
    def test_byte_hash_deterministic(self):
        img = np.random.RandomState(0).randint(0, 256, (64, 64, 3),
                                                dtype=np.uint8)
        self.assertEqual(byte_hash(img), byte_hash(img.copy()))

    def test_phash_similar_frames(self):
        img = np.random.RandomState(0).randint(0, 256, (64, 64, 3),
                                                dtype=np.uint8)
        img2 = img.copy()
        img2[0, 0] = np.uint8((int(img2[0, 0, 0]) + 1) % 256)  # tiny perturbation
        self.assertLessEqual(hamming(phash(img), phash(img2)), 5)


class TestCrossTaskCache(unittest.TestCase):
    def test_hit_miss_accounting(self):
        c = CrossTaskCache()
        img = np.random.RandomState(1).randint(0, 256, (32, 32, 3),
                                                dtype=np.uint8)
        fid = FrameIdentity(img)
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return np.array([1.0, 2.0])

        v1, hit = c.get_or_compute(fid, "m1", compute)
        v2, hit2 = c.get_or_compute(fid, "m1", compute)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(hit)
        self.assertTrue(hit2)
        np.testing.assert_array_equal(v1, v2)

    def test_model_isolation(self):
        c = CrossTaskCache()
        img = np.ones((32, 32, 3), dtype=np.uint8)
        fid = FrameIdentity(img)
        c.put(fid, "m1", 10)
        c.put(fid, "m2", 20)
        self.assertEqual(c.get(fid, "m1"), 10)
        self.assertEqual(c.get(fid, "m2"), 20)


class TestSamplers(unittest.TestCase):
    def test_uniform_budget(self):
        s = UniformSampler(fps=2.0, max_samples=10)
        out = s.sample("dummy.mp4", duration=8.0)
        self.assertLessEqual(len(out), 10)
        self.assertEqual(out[0][0], 0.0)

    def test_hybrid_dedup(self):
        s = HybridSampler([
            UniformSampler(fps=2.0),
            UniformSampler(fps=4.0),
        ], dedup_gap_sec=0.1)
        out = s.sample("dummy.mp4", duration=2.0)
        ts = [t for t, _ in out]
        for i in range(1, len(ts)):
            self.assertGreaterEqual(ts[i] - ts[i - 1], 0.09)


class TestBuildProtos(unittest.TestCase):
    def test_shapes(self):
        x = np.random.randn(10, 8).astype(np.float32)
        p = build_protos(x, K=4)
        self.assertEqual(p.shape, (4, 8))
        # unit-norm rows
        norms = np.linalg.norm(p, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_small_n(self):
        x = np.random.randn(2, 8).astype(np.float32)
        p = build_protos(x, K=6)
        self.assertEqual(p.shape, (6, 8))


class TestOfflineIndexInMemory(unittest.TestCase):
    def test_search_trivial(self):
        # Build 3 entries with deterministic embeddings; query matches #2.
        d = 8
        def mk(vec):
            v = vec / (np.linalg.norm(vec) + 1e-9)
            return np.tile(v, (5, 1)).astype(np.float32)
        e0 = VideoIndexEntry(
            video_id="v0", video_path="", duration=1.0,
            frame_embs=mk(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
            protos={2: build_protos(
                mk(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)), 2
            )},
        )
        e1 = VideoIndexEntry(
            video_id="v1", video_path="", duration=1.0,
            frame_embs=mk(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
            protos={2: build_protos(
                mk(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)), 2
            )},
        )
        e2 = VideoIndexEntry(
            video_id="v2", video_path="", duration=1.0,
            frame_embs=mk(np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)),
            protos={2: build_protos(
                mk(np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)), 2
            )},
        )
        idx = OfflineIndex([e0, e1, e2])
        q = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        # For trivial 3-video test, disable NNN/QAMP/col-softmax so only
        # base cosine drives the ranking.
        hits = idx.search(q, top_k=3, col_beta=0.0, alpha_nnn=0.0,
                          topm_rerank=1)
        self.assertEqual(hits[0][0], "v2")

    def test_persistence(self):
        d = 8
        e = VideoIndexEntry(
            video_id="v0", video_path="", duration=1.0,
            frame_embs=np.eye(d, dtype=np.float32),
            protos={2: np.eye(2, d, dtype=np.float32)},
        )
        idx = OfflineIndex([e])
        with tempfile.TemporaryDirectory() as d_:
            p = Path(d_) / "idx.pkl"
            idx.save(str(p))
            loaded = OfflineIndex.load(str(p))
            self.assertEqual(loaded.size, 1)
            self.assertEqual(loaded.entries[0].video_id, "v0")


class TestQueryPlanner(unittest.TestCase):
    def _hits(self, margin):
        # top-1=0.8, top-2=0.8-margin
        return [("v1", 0.8, margin), ("v2", 0.8 - margin, margin),
                ("v3", 0.5, margin)]

    def test_easy(self):
        p = QueryPlanner(QueryPlannerConfig(easy_margin=0.1, hard_margin=0.02))
        plan = p.plan(self._hits(0.15))
        self.assertEqual(plan.difficulty, QueryDifficulty.EASY)
        self.assertFalse(plan.run_momentdetr)

    def test_medium(self):
        p = QueryPlanner(QueryPlannerConfig(easy_margin=0.1, hard_margin=0.02))
        plan = p.plan(self._hits(0.05))
        self.assertEqual(plan.difficulty, QueryDifficulty.MEDIUM)
        self.assertFalse(plan.run_momentdetr)

    def test_hard(self):
        p = QueryPlanner(QueryPlannerConfig(easy_margin=0.1, hard_margin=0.02))
        plan = p.plan(self._hits(0.001))
        self.assertEqual(plan.difficulty, QueryDifficulty.HARD)
        self.assertTrue(plan.run_momentdetr)


if __name__ == "__main__":
    unittest.main()
