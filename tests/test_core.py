"""Unit tests — run via `python -m unittest discover tests` from project root."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.scheduler import UnifiedScheduler
from core.subscription import TaskSubscription
from core.two_stage import TwoStageController
from core.types import InterestSignal, Interval, SamplingStage
from core.cache import SharedFrameCache
from core.segment_aggregator import Segment, SegmentAggregator, segments_mean_iou, boundary_mae


class TestScheduler(unittest.TestCase):
    def test_plan_sparse_single_task(self):
        sub = TaskSubscription(task_id="t1", sparse_fps=1.0,
                               can_produce_interest=True)
        sched = UnifiedScheduler([sub])
        reqs = sched.plan_sparse("v", duration=5.0, prefilter=None)
        self.assertEqual(len(reqs), 5)
        self.assertTrue(all("t1" in r.subscribers for r in reqs))
        self.assertTrue(all(r.stage == SamplingStage.SPARSE for r in reqs))

    def test_plan_sparse_multi_task_dedup(self):
        a = TaskSubscription(task_id="a", sparse_fps=1.0,
                             can_produce_interest=True)
        b = TaskSubscription(task_id="b", sparse_fps=1.0)
        sched = UnifiedScheduler([a, b])
        reqs = sched.plan_sparse("v", duration=5.0, prefilter=None)
        # Both tasks sample at the same timestamps -> dedup
        self.assertEqual(len(reqs), 5)
        for r in reqs:
            self.assertSetEqual(r.subscribers, {"a", "b"})

    def test_gated_task_skipped_in_sparse(self):
        a = TaskSubscription(task_id="det", sparse_fps=1.0,
                             can_produce_interest=True)
        b = TaskSubscription(task_id="emb", sparse_fps=1.0,
                             gated_by="det")
        sched = UnifiedScheduler([a, b])
        reqs = sched.plan_sparse("v", duration=3.0, prefilter=None)
        for r in reqs:
            self.assertNotIn("emb", r.subscribers)

    def test_plan_dense_only_full_tasks(self):
        a = TaskSubscription(task_id="a", sparse_fps=1.0, dense_fps=2.0,
                             can_produce_interest=True)
        b = TaskSubscription(task_id="b", sparse_fps=1.0)  # not full-path
        c = TaskSubscription(task_id="c", sparse_fps=0.0, dense_fps=1.0,
                             gated_by="a")
        sched = UnifiedScheduler([a, b, c])
        intervals = [Interval(start=1.0, end=2.0, score=1.0)]
        reqs = sched.plan_dense("v", intervals, prefilter=None)
        # Only 'a' and 'c' should appear
        all_subs = set()
        for r in reqs:
            all_subs |= r.subscribers
        self.assertIn("a", all_subs)
        self.assertIn("c", all_subs)
        self.assertNotIn("b", all_subs)


class TestTwoStageController(unittest.TestCase):
    def test_aggregate_merges_overlapping(self):
        ctl = TwoStageController(interval_expand_sec=0.0, merge_gap_sec=0.5)
        sigs = [
            InterestSignal("t1", [
                Interval(1.0, 2.0, score=0.8),
                Interval(2.3, 3.0, score=0.7),
            ]),
            InterestSignal("t2", [
                Interval(1.5, 2.5, score=0.9),
            ]),
        ]
        out = ctl.aggregate(sigs)
        # The three intervals are within merge_gap -> collapse to one
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].start, 1.0)
        self.assertAlmostEqual(out[0].end, 3.0)

    def test_aggregate_respects_max_intervals(self):
        ctl = TwoStageController(max_intervals=2, merge_gap_sec=0.0,
                                  interval_expand_sec=0.0)
        sigs = [
            InterestSignal("t", [
                Interval(0.0, 0.5, score=0.9),
                Interval(2.0, 2.5, score=0.8),
                Interval(4.0, 4.5, score=0.7),
            ]),
        ]
        out = ctl.aggregate(sigs)
        self.assertEqual(len(out), 2)


class TestCache(unittest.TestCase):
    def test_lru_eviction(self):
        c = SharedFrameCache(max_size=2)
        c.put(1, np.zeros((4, 4, 3), dtype=np.uint8))
        c.put(2, np.zeros((4, 4, 3), dtype=np.uint8))
        c.put(3, np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertIsNone(c.get(1))
        self.assertIsNotNone(c.get(2))
        self.assertIsNotNone(c.get(3))


class TestSegmentAggregator(unittest.TestCase):
    def test_empty_input(self):
        agg = SegmentAggregator()
        self.assertEqual(agg.aggregate([]), [])

    def test_single_peak(self):
        # One strong peak surrounded by low scores
        pairs = [(float(i), 0.1) for i in range(20)]
        for i in range(8, 13):
            pairs[i] = (float(i), 0.9)
        agg = SegmentAggregator(percentile=0.70, smooth_window=1,
                                merge_gap_sec=1.0, min_segment_sec=0.0)
        segs = agg.aggregate(pairs)
        self.assertGreaterEqual(len(segs), 1)
        # The segment should cover roughly t=8..12
        seg = segs[0]
        self.assertLessEqual(seg.start, 8.0)
        self.assertGreaterEqual(seg.end, 12.0)
        self.assertGreater(seg.score, 0.5)

    def test_merge_close_segments(self):
        # Two clusters with a tiny gap
        pairs = [(0.0, 0.9), (1.0, 0.9), (2.0, 0.1),
                 (3.0, 0.9), (4.0, 0.9)]
        agg = SegmentAggregator(percentile=0.50, smooth_window=1,
                                merge_gap_sec=2.0, min_segment_sec=0.0)
        segs = agg.aggregate(pairs)
        # Should merge into one segment since gap (1.0) < merge_gap (2.0)
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0].start, 0.0)
        self.assertAlmostEqual(segs[0].end, 4.0)

    def test_min_length_filter(self):
        # Single isolated peak produces a zero-length segment
        pairs = [(0.0, 0.1), (1.0, 0.1), (2.0, 0.9), (3.0, 0.1), (4.0, 0.1)]
        agg = SegmentAggregator(absolute_threshold=0.5, smooth_window=1,
                                merge_gap_sec=0.0, min_segment_sec=2.0)
        segs = agg.aggregate(pairs)
        # The peak at t=2.0 produces a zero-length segment, filtered by min_len=2.0
        self.assertEqual(len(segs), 0)

    def test_max_segments(self):
        pairs = [(float(i * 10), 0.9) for i in range(20)]
        agg = SegmentAggregator(percentile=0.10, smooth_window=1,
                                merge_gap_sec=0.0, min_segment_sec=0.0,
                                max_segments=3)
        segs = agg.aggregate(pairs)
        self.assertLessEqual(len(segs), 3)

    def test_absolute_threshold(self):
        pairs = [(0.0, 0.3), (1.0, 0.7), (2.0, 0.8), (3.0, 0.2)]
        agg = SegmentAggregator(absolute_threshold=0.5, smooth_window=1,
                                merge_gap_sec=0.5, min_segment_sec=0.0)
        segs = agg.aggregate(pairs)
        self.assertGreaterEqual(len(segs), 1)
        # The segment should cover t=1..2
        self.assertLessEqual(segs[0].start, 1.0)
        self.assertGreaterEqual(segs[0].end, 2.0)

    def test_nms_removes_overlapping(self):
        agg = SegmentAggregator(nms_iou=0.3)
        seg_a = Segment(0.0, 5.0, 0.9)
        seg_b = Segment(1.0, 6.0, 0.8)  # high overlap with a
        seg_c = Segment(20.0, 25.0, 0.7)
        kept = agg._nms([seg_a, seg_b, seg_c], iou_thr=0.3)
        # seg_b overlaps significantly with seg_a, should be dropped
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(kept[0].start, 0.0)
        self.assertAlmostEqual(kept[1].start, 20.0)


class TestSegmentMetrics(unittest.TestCase):
    def test_perfect_iou(self):
        segs = [Segment(0.0, 5.0, 1.0), Segment(10.0, 15.0, 1.0)]
        self.assertAlmostEqual(segments_mean_iou(segs, segs), 1.0)

    def test_zero_iou(self):
        pred = [Segment(0.0, 1.0, 1.0)]
        gt = [Segment(10.0, 11.0, 1.0)]
        self.assertAlmostEqual(segments_mean_iou(pred, gt), 0.0)

    def test_both_empty(self):
        self.assertAlmostEqual(segments_mean_iou([], []), 1.0)
        self.assertAlmostEqual(boundary_mae([], []), 0.0)

    def test_boundary_mae_perfect(self):
        segs = [Segment(1.0, 5.0, 1.0)]
        self.assertAlmostEqual(boundary_mae(segs, segs), 0.0)

    def test_boundary_mae_offset(self):
        pred = [Segment(1.5, 5.5, 1.0)]
        gt = [Segment(1.0, 5.0, 1.0)]
        # |1.5-1.0| + |5.5-5.0| = 1.0
        self.assertAlmostEqual(boundary_mae(pred, gt), 1.0)


if __name__ == "__main__":
    unittest.main()
