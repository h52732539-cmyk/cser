"""Phase 3: metadata-aware retrieval tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.metadata import (
    VideoMetadata, classify_motion_from_sensor, classify_geo,
    fill_derived_fields,
)
from core.query_parser import QueryParser, QueryIntent
from core.meta_filter import MetaFilter, fuse_scores
from core.offline_index import OfflineIndex, VideoIndexEntry, build_protos


# ----------------------------------------------------------------------

class TestMotionClassifier(unittest.TestCase):
    def test_stationary(self):
        g = np.zeros((100, 3))
        a = np.array([[0, 0, 9.8]] * 100)
        r = classify_motion_from_sensor({"gyro": g, "gyro_fps": 50,
                                          "accel": a, "accel_fps": 50})
        self.assertEqual(r["class"], "stationary")

    def test_running_like(self):
        # 3 Hz vertical oscillation of strong amplitude
        t = np.linspace(0, 2.0, 100)
        verts = 3.0 * np.sin(2 * np.pi * 3.0 * t)
        a = np.stack([np.zeros_like(t), np.zeros_like(t), 9.8 + verts],
                     axis=-1)
        g = 0.3 * np.random.randn(100, 3)
        r = classify_motion_from_sensor({"gyro": g, "gyro_fps": 50,
                                          "accel": a, "accel_fps": 50})
        self.assertIn(r["class"], ("running", "walking"))


class TestGeoClassifier(unittest.TestCase):
    def test_hainan_coast(self):
        self.assertEqual(classify_geo(19.0, 109.5), "coast")

    def test_unknown_inland(self):
        self.assertEqual(classify_geo(30.0, 103.0), "unknown")

    def test_mountain(self):
        self.assertEqual(classify_geo(30.0, 103.0, alt=1500), "mountain")

    def test_none_inputs(self):
        self.assertEqual(classify_geo(None, None), "unknown")


# ----------------------------------------------------------------------

class TestQueryParser(unittest.TestCase):
    def setUp(self):
        # fix "now" so time tests are deterministic (2026-04-24 00:00 UTC)
        from datetime import datetime, timezone
        self.now = datetime(2026, 4, 24, 0, 0, tzinfo=timezone.utc).timestamp()
        self.p = QueryParser(now_ts=self.now)

    def test_zh_coast_running(self):
        it = self.p.parse("上周末在海边跑步的视频")
        self.assertIn("coast",   it.geo_categories)
        self.assertIn("running", it.motion_classes)
        self.assertIsNotNone(it.time_window)
        s, e = it.time_window
        self.assertLessEqual(s, self.now)

    def test_en_last_week_walking(self):
        it = self.p.parse("walking on the beach last week")
        self.assertIn("coast",   it.geo_categories)
        self.assertIn("walking", it.motion_classes)
        self.assertIsNotNone(it.time_window)

    def test_year(self):
        it = self.p.parse("videos from 2024")
        self.assertIsNotNone(it.time_window)
        s, e = it.time_window
        from datetime import datetime, timezone
        self.assertAlmostEqual(s,
            datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp(), places=0)

    def test_no_constraint(self):
        it = self.p.parse("a running dog")
        # "running" → motion, no time, no geo
        self.assertEqual(it.time_window, None)
        self.assertEqual(it.geo_categories, [])
        self.assertIn("running", it.motion_classes)


# ----------------------------------------------------------------------

def _mk_entry(vid, emb, meta):
    e = VideoIndexEntry(
        video_id=vid, video_path="", duration=10.0,
        frame_embs=np.tile(emb / (np.linalg.norm(emb) + 1e-9),
                            (4, 1)).astype(np.float32),
        protos={2: build_protos(
            np.tile(emb / (np.linalg.norm(emb) + 1e-9),
                    (4, 1)).astype(np.float32), 2)},
        metadata=meta,
    )
    return e


class TestMetaFilter(unittest.TestCase):
    def setUp(self):
        # 3 videos: [coast/running at t=100], [mountain/walking at t=200],
        #           [unknown at t=300]
        m1 = VideoMetadata(creation_time=100.0, latitude=19.0,
                            longitude=109.5, motion_class="running",
                            motion_confidence=0.9,
                            geo_category="coast")
        m2 = VideoMetadata(creation_time=200.0, motion_class="walking",
                            motion_confidence=0.8,
                            geo_category="mountain")
        m3 = VideoMetadata(creation_time=300.0)
        self.metas = [m1, m2, m3]

    def test_filter_geo(self):
        mf = MetaFilter()
        it = QueryIntent(semantic_text="", geo_categories=["coast"])
        fr = mf.filter(self.metas, it)
        self.assertEqual(fr.mask.tolist(), [True, False, True])
        # m3 has no geo_category so non-strict keeps it

    def test_filter_strict(self):
        mf = MetaFilter(strict=True)
        it = QueryIntent(semantic_text="", geo_categories=["coast"])
        fr = mf.filter(self.metas, it)
        self.assertEqual(fr.mask.tolist(), [True, False, False])

    def test_filter_time(self):
        mf = MetaFilter(time_slack_sec=0.0)
        it = QueryIntent(semantic_text="", time_window=(150.0, 250.0))
        fr = mf.filter(self.metas, it)
        self.assertEqual(fr.mask.tolist(), [False, True, False])

    def test_soft_score(self):
        mf = MetaFilter()
        it = QueryIntent(semantic_text="",
                          geo_categories=["coast"],
                          motion_classes=["running"])
        s = mf.soft_score(self.metas, it)
        # m1 should score highest (matches both)
        self.assertGreater(s[0], s[1])
        self.assertGreater(s[0], s[2])

    def test_no_constraint(self):
        mf = MetaFilter()
        it = QueryIntent(semantic_text="")
        s = mf.soft_score(self.metas, it)
        np.testing.assert_array_equal(s, np.ones(3, dtype=np.float32))


class TestSearchWithMeta(unittest.TestCase):
    def setUp(self):
        self.D = 8
        def _emb(i):
            v = np.zeros(self.D, dtype=np.float32); v[i] = 1.0; return v
        self.entries = [
            _mk_entry("v0", _emb(0),
                VideoMetadata(creation_time=100, geo_category="coast",
                              motion_class="running",
                              motion_confidence=0.9)),
            _mk_entry("v1", _emb(1),
                VideoMetadata(creation_time=200, geo_category="mountain",
                              motion_class="walking",
                              motion_confidence=0.9)),
            _mk_entry("v2", _emb(2),
                VideoMetadata(creation_time=300, geo_category="urban")),
        ]

    def test_basic_hybrid(self):
        idx = OfflineIndex(entries=self.entries)
        # query that should prefer v0 (coast+running)
        q = np.zeros(self.D, dtype=np.float32); q[1] = 1.0
        it = QueryIntent(semantic_text="", geo_categories=["coast"],
                          motion_classes=["running"])
        hits = idx.search_with_meta(q, it, top_k=3)
        # After filter, v1 and v2 should be excluded (strict=False keeps
        # them only if meta is missing; v1 is mountain so excluded).
        kept_ids = [h[0] for h in hits if h[1] > -1e8]
        self.assertIn("v0", kept_ids[:1])

    def test_fallback_no_constraint(self):
        idx = OfflineIndex(entries=self.entries)
        q = np.zeros(self.D, dtype=np.float32); q[1] = 1.0
        it = QueryIntent(semantic_text="")  # no meta constraint
        # NNN/col-softmax on a 1-query batch is degenerate (std=0 across
        # the single column); disable them so the trivial test reduces
        # to pure cosine.
        hits = idx.search_with_meta(
            q, it, top_k=3,
            alpha_nnn=0.0, tau_qamp=1e9, col_beta=0.0, topm_rerank=1,
        )
        # With no constraints, top-1 should be v1 (matches q dimension)
        self.assertEqual(hits[0][0], "v1")


# ----------------------------------------------------------------------

class TestFuseScores(unittest.TestCase):
    def test_blend(self):
        sem  = np.array([0.9, 0.1, 0.5], dtype=np.float32)
        meta = np.array([0.0, 1.0, 0.5], dtype=np.float32)
        f = fuse_scores(sem, meta, alpha=0.5)
        # sem gets normalized to [0,1]: [1.0, 0.0, 0.5]
        # fused = 0.5*[1,0,0.5] + 0.5*[0,1,0.5] = [0.5, 0.5, 0.5]
        np.testing.assert_allclose(f, [0.5, 0.5, 0.5], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
