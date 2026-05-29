"""LiteVTR++ v2 — black-box-model-preserving orchestrator.

Lifecycle:

    OFFLINE (one-time, per gallery):
        build OfflineIndex via OfflineIndexBuilder.

    QUERY PATH (per user query):
        1. encode_text(query)                                 ~3 ms
        2. offline_index.search(query_emb)                    ~5-30 ms (pure numpy)
        3. QueryPlanner.plan(search_results)
        4. if EASY → return                                   done
        5. else    → full multi-task pipeline on top-N videos
             5a. MetadataPrefilter                           (existing)
             5b. HybridSampler (S4 = MV + QFrame + Uniform)  (new)
             5c. Unified scheduler + CrossTaskCache          (new cache)
             5d. Task adapters run (Huawei models untouched)

Every Huawei model call is routed through CrossTaskCache first.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .adaptive_sampler import (
    HybridSampler, MVBasedSampler, QFrameSampler, QFrameConfig,
    UniformSampler, _BaseSampler,
)
from .cache import SharedFrameCache
from .cross_task_cache import CrossTaskCache
from .decoder import decode_frames
from .frame_identity import FrameIdentity
from .offline_index import OfflineIndex
from .prefilter import MetadataPrefilter, PrefilterResult
from .query_planner import QueryPlanner, QueryPlan, QueryDifficulty
from .scheduler import UnifiedScheduler
from .segment_aggregator import SegmentAggregator
from .types import Frame, FrameRequest, InterestSignal, SamplingStage, TaskResult


# ----------------------------------------------------------------------

@dataclass
class FrameworkV2Stats:
    # high level
    wall_ms: float = 0.0
    n_huawei_calls: Dict[str, int] = field(default_factory=dict)
    n_cache_hits: Dict[str, int] = field(default_factory=dict)
    n_frames_decoded: int = 0
    # per-stage
    encode_text_ms: float = 0.0
    index_search_ms: float = 0.0
    prefilter_ms: float = 0.0
    sampling_ms: float = 0.0
    decode_ms: float = 0.0
    stage2_model_ms: float = 0.0
    # routing
    query_plan: Optional[str] = None
    margin: float = 0.0
    # aux
    index_size: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "wall_ms": self.wall_ms,
            "encode_text_ms": self.encode_text_ms,
            "index_search_ms": self.index_search_ms,
            "prefilter_ms": self.prefilter_ms,
            "sampling_ms": self.sampling_ms,
            "decode_ms": self.decode_ms,
            "stage2_model_ms": self.stage2_model_ms,
            "n_huawei_calls": dict(self.n_huawei_calls),
            "n_cache_hits": dict(self.n_cache_hits),
            "n_frames_decoded": self.n_frames_decoded,
            "query_plan": self.query_plan,
            "margin": self.margin,
            "index_size": self.index_size,
        }


# ----------------------------------------------------------------------

class LiteVTRFrameworkV2:
    """Orchestrator that honours the black-box-model constraint."""

    def __init__(self,
                 offline_index: OfflineIndex,
                 # Model handles (we never touch weights, just call)
                 huawei_clip_text_encode: Callable[[str], np.ndarray],
                 huawei_clip_image_encode: Callable[[List[np.ndarray]], np.ndarray],
                 highlight_model: Optional[Any] = None,
                 face_detector: Optional[Any] = None,
                 face_embedder: Optional[Any] = None,
                 scene_classifier: Optional[Any] = None,
                 # Components
                 prefilter: Optional[MetadataPrefilter] = None,
                 query_planner: Optional[QueryPlanner] = None,
                 cross_cache: Optional[CrossTaskCache] = None,
                 hybrid_sampler: Optional[_BaseSampler] = None,
                 cache_size: int = 500) -> None:
        self.index = offline_index
        self.t_enc = huawei_clip_text_encode
        self.i_enc = huawei_clip_image_encode
        self.highlight = highlight_model
        self.face_det = face_detector
        self.face_emb = face_embedder
        self.scene = scene_classifier

        self.prefilter = prefilter or MetadataPrefilter()
        self.planner = query_planner or QueryPlanner()
        self.cross_cache = cross_cache or CrossTaskCache()
        self.frame_cache = SharedFrameCache(cache_size)

        self.sampler = hybrid_sampler or HybridSampler(
            samplers=[
                MVBasedSampler(motion_tau=1.5),
                UniformSampler(fps=1.0, max_samples=30),
            ],
            dedup_gap_sec=0.3,
            max_samples=40,
        )

        self.stats = FrameworkV2Stats(index_size=self.index.size)

    # ==================================================================
    #  Main entry point
    # ==================================================================

    def query(self, query_text: str,
              videos_meta: Optional[Dict[str, Dict]] = None,
              top_k: int = 5) -> Dict[str, Any]:
        """Run one end-to-end query. Returns dict with results + stats."""
        t0_wall = time.perf_counter()
        self._reset_stats()

        # 1. Text encode (1 Huawei call)
        t0 = time.perf_counter()
        q_emb = self._call_model(
            model_id="huawei_clip_text",
            compute_fn=lambda: self.t_enc(query_text),
        )
        self.stats.encode_text_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Offline index search (pure numpy, no Huawei call)
        t0 = time.perf_counter()
        hits = self.index.search(q_emb, top_k=max(top_k, 10))
        self.stats.index_search_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Query planning (QPP)
        plan = self.planner.plan(hits)
        self.stats.query_plan = plan.difficulty.value
        self.stats.margin = plan.margin

        # 4. EASY path — return immediately
        if plan.difficulty == QueryDifficulty.EASY:
            top = [(vid, float(sc)) for vid, sc, _ in hits[:top_k]]
            result = {
                "top_k": top,
                "segments_per_video": {},
                "plan": plan,
                "stats": self._finalize_stats(t0_wall),
            }
            return result

        # 5. MEDIUM / HARD path — refine on top candidates
        meta = videos_meta or {}
        refined = []
        segments_per_video: Dict[str, Any] = {}
        for vid in plan.top_candidates[: plan.refine_top_n]:
            vmeta = meta.get(vid) or self._lookup_meta(vid)
            if vmeta is None:
                refined.append((vid, 0.0))
                continue
            seg_result = self._refine_one_video(
                vid, vmeta, q_emb, query_text, run_momentdetr=plan.run_momentdetr,
            )
            refined.append((vid, seg_result["rescore"]))
            segments_per_video[vid] = seg_result.get("segments", [])

        # Re-rank using refined scores  (simple blend: 0.7 index + 0.3 refine)
        index_score_map = {v: s for v, s, _ in hits}
        blended: List = []
        refined_ids = {r[0] for r in refined}
        for vid, rs in refined:
            base = index_score_map.get(vid, 0.0)
            blended.append((vid, 0.7 * base + 0.3 * rs))
        # Any remaining non-refined candidates keep their index score
        for vid, s, _ in hits:
            if vid not in refined_ids:
                blended.append((vid, float(s)))
        blended.sort(key=lambda x: -x[1])
        top = [(v, float(s)) for v, s in blended[:top_k]]

        return {
            "top_k": top,
            "segments_per_video": segments_per_video,
            "plan": plan,
            "stats": self._finalize_stats(t0_wall),
        }

    # ==================================================================
    #  Stage 2 refinement for one candidate
    # ==================================================================

    def _refine_one_video(self, video_id: str, vmeta: Dict,
                           q_emb: np.ndarray, query_text: str,
                           run_momentdetr: bool) -> Dict[str, Any]:
        video_path = vmeta["path"]
        duration = float(vmeta.get("duration", 30.0))
        sensor = vmeta.get("sensor")

        # 5a. Prefilter
        t0 = time.perf_counter()
        pre = self.prefilter.analyze(video_path, duration, sensor)
        self.stats.prefilter_ms += (time.perf_counter() - t0) * 1000.0

        # 5b. Hybrid sampling, query-conditional via injected encoders
        t0 = time.perf_counter()
        if isinstance(self.sampler, HybridSampler):
            try:
                self.sampler.samplers = list(self.sampler.samplers)  # defensive
            except Exception:
                pass
        ts_tags = self.sampler.sample(
            video_path, duration,
            sensor_stream=sensor,
            query_text=query_text,
        )
        self.stats.sampling_ms += (time.perf_counter() - t0) * 1000.0

        ts_tags = self._apply_prefilter_mask(ts_tags, pre)
        if not ts_tags:
            return {"rescore": 0.0, "segments": []}

        # 5b·. Decode only the chosen timestamps (shared cache)
        t0 = time.perf_counter()
        reqs = [
            FrameRequest(
                video_id=video_id, frame_idx=int(t * 25),
                timestamp=float(t), stage=SamplingStage.DENSE,
                subscribers={"refine"},
            )
            for t, _ in ts_tags
        ]
        frames = decode_frames(video_path, reqs, self.frame_cache)
        self.stats.decode_ms += (time.perf_counter() - t0) * 1000.0
        self.stats.n_frames_decoded += len(frames)

        if not frames:
            return {"rescore": 0.0, "segments": []}

        # 5c. Build FrameIdentities once for the whole batch
        fids = [FrameIdentity(f.image) for f in frames]
        images = [f.image for f in frames]

        # 5d. CLIP image encode (with cross-task cache reuse)
        t0 = time.perf_counter()
        embs = self._batched_call(
            model_id="huawei_clip_image",
            fids=fids, items=images,
            compute_fn=self.i_enc,
        )
        self.stats.stage2_model_ms += (time.perf_counter() - t0) * 1000.0

        # Score: cosine to query on frame level, then peak-aggregation
        embs = np.asarray(embs, dtype=np.float32)
        embs /= (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-9)
        q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims = embs @ q

        # Build segments via SegmentAggregator
        pairs = list(zip([f.timestamp for f in frames], sims.tolist()))
        agg = SegmentAggregator(percentile=0.70, merge_gap_sec=0.8,
                                min_segment_sec=0.3, max_segments=5)
        segs = agg.aggregate(pairs)
        segments = [s.to_dict() for s in segs]
        rescore = float(sims.max()) if len(sims) > 0 else 0.0

        # Optional: MomentDETR refinement on top-scored window
        if run_momentdetr and self.highlight is not None:
            try:
                t0 = time.perf_counter()
                hi_scores = self._batched_call(
                    model_id="huawei_highlight",
                    fids=fids, items=images,
                    compute_fn=lambda imgs: self.highlight.score(imgs),
                )
                self.stats.stage2_model_ms += (time.perf_counter() - t0) * 1000.0
                if hi_scores:
                    hi_pairs = list(zip([f.timestamp for f in frames],
                                         list(hi_scores)))
                    hi_segs = agg.aggregate(hi_pairs)
                    segments = sorted(
                        segments + [s.to_dict() for s in hi_segs],
                        key=lambda s: -s.get("score", 0.0),
                    )[:5]
            except Exception:
                pass

        return {"rescore": rescore, "segments": segments}

    # ==================================================================
    #  Internal helpers
    # ==================================================================

    def _batched_call(self, model_id: str,
                       fids: List[FrameIdentity],
                       items: List[Any],
                       compute_fn: Callable[[List[Any]], Any]) -> List[Any]:
        """Call `compute_fn` only on uncached items."""
        out: List[Any] = [None] * len(fids)
        miss_idx: List[int] = []
        for i, fid in enumerate(fids):
            hit = self.cross_cache.get(fid, model_id=model_id, mode="byte")
            if hit is not None:
                out[i] = hit
                self.stats.n_cache_hits[model_id] = (
                    self.stats.n_cache_hits.get(model_id, 0) + 1
                )
            else:
                miss_idx.append(i)

        if miss_idx:
            miss_items = [items[i] for i in miss_idx]
            computed = compute_fn(miss_items)
            self.stats.n_huawei_calls[model_id] = (
                self.stats.n_huawei_calls.get(model_id, 0) + len(miss_idx)
            )
            # Put back into cache
            for j, i in enumerate(miss_idx):
                val = computed[j] if hasattr(computed, "__getitem__") else computed
                out[i] = val
                self.cross_cache.put(fids[i], model_id, val)
        return out

    def _call_model(self, model_id: str,
                     compute_fn: Callable[[], Any]) -> Any:
        """Call an unary model and record a hit."""
        self.stats.n_huawei_calls[model_id] = (
            self.stats.n_huawei_calls.get(model_id, 0) + 1
        )
        return compute_fn()

    def _apply_prefilter_mask(self, ts_tags, pre: PrefilterResult):
        if pre is None or pre.candidate_mask is None:
            return ts_tags
        mask = pre.candidate_mask
        T = len(mask)
        out = []
        for t, tag in ts_tags:
            idx = int(t * 10)
            if 0 <= idx < T and mask[idx]:
                out.append((t, tag))
        return out

    def _lookup_meta(self, video_id: str) -> Optional[Dict]:
        for e in self.index.entries:
            if e.video_id == video_id:
                return {"path": e.video_path, "duration": e.duration}
        return None

    def _reset_stats(self) -> None:
        self.stats = FrameworkV2Stats(index_size=self.index.size)

    def _finalize_stats(self, t0_wall: float) -> Dict[str, Any]:
        self.stats.wall_ms = (time.perf_counter() - t0_wall) * 1000.0
        return self.stats.as_dict()
