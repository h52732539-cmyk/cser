"""Sampling-pipeline ablation — modules that affect the *decode/sample/
compute* pipeline (not retrieval scoring).

Operates on real MSR-VTT videos and measures end-to-end wall_ms,
n_frames_decoded, n_huawei_calls under each toggle.

Toggles:
    S0  full pipeline
    S1  no MetadataPrefilter
    S2  no two-stage feedback (forces uniform-only sampling, single pass)
    S3  no UnifiedScheduler  (each task decodes independently)
    S4  no SegmentAggregator
    S5  no Adaptive Sampler  (uniform 2 fps fallback)
    S6  no CrossTaskCache    (re-run every model per call across stages)
    S7  no QPP routing       (always full Stage-2)

Each toggle measures (averaged over N videos, after warmup):
    n_frames_decoded
    n_huawei_model_calls (per model)
    cache_hits / misses
    wall_ms

Important methodology fixes vs earlier version:
  * dry-run warmup before measurement to remove first-call ONNX latency
  * two-stage flow (sparse 1 fps → dense 2 fps) so CrossTaskCache has
    meaningful sparse↔dense reuse opportunity
  * separate "wall_ms_compute" (model time only) vs "wall_ms_total"
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.types import FrameRequest, SamplingStage
from core.decoder import decode_frames, probe_video
from core.adaptive_sampler import (
    HybridSampler, MVBasedSampler, UniformSampler,
)
from core.frame_identity import FrameIdentity
from core.cross_task_cache import CrossTaskCache
from core.prefilter import MetadataPrefilter
from core.segment_aggregator import SegmentAggregator

from tasks.real_models import (
    MomentDETRHighlightModel, InsightFaceDetector, InsightFaceEmbedder,
    MobileNetV3SceneClassifier,
)


@dataclass
class SamplingConfig:
    name: str = "S0_full"
    enable_prefilter:        bool = True
    enable_two_stage:        bool = True
    enable_unified_scheduler: bool = True
    enable_segment_aggregator: bool = True
    enable_adaptive_sampler: bool = True
    enable_cross_cache:      bool = True
    enable_qpp_routing:      bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


# ----------------------------------------------------------------------

def _decode_at(video_path: str, video_id: str, timestamps: List[float]):
    reqs = [FrameRequest(video_id=video_id, frame_idx=int(t * 25),
                          timestamp=float(t), stage=SamplingStage.DENSE,
                          subscribers={"all"}) for t in timestamps]
    return decode_frames(video_path, reqs)


def _sample_timestamps(video_path: str, duration: float,
                        cfg: SamplingConfig,
                        target_fps: float = 2.0) -> Dict[str, List[float]]:
    """Return dict with 'sparse' and 'dense' timestamps."""
    if cfg.enable_two_stage:
        # Sparse @ 1 fps, dense @ target_fps
        sparse_ts = list(np.arange(0, duration, 1.0))
        if cfg.enable_adaptive_sampler:
            target_n = max(int(duration * target_fps), 30)
            sampler = HybridSampler(
                samplers=[
                    MVBasedSampler(motion_tau=1.0, max_samples=target_n),
                    UniformSampler(fps=target_fps, max_samples=target_n),
                ],
                dedup_gap_sec=0.15,
                max_samples=target_n + 20,
            )
            ts_tags = sampler.sample(video_path, duration)
            dense_ts = [t for t, _ in ts_tags]
        else:
            dense_ts = list(np.arange(0, duration, 1.0 / target_fps))
        return {"sparse": sparse_ts, "dense": dense_ts}
    else:
        # No two-stage: single combined set
        if cfg.enable_adaptive_sampler:
            target_n = max(int(duration * target_fps), 30)
            sampler = HybridSampler(
                samplers=[
                    MVBasedSampler(motion_tau=1.0, max_samples=target_n),
                    UniformSampler(fps=target_fps, max_samples=target_n),
                ],
                dedup_gap_sec=0.15,
                max_samples=target_n + 20,
            )
            ts_tags = sampler.sample(video_path, duration)
            ts = [t for t, _ in ts_tags]
        else:
            ts = list(np.arange(0, duration, 1.0 / target_fps))
        return {"sparse": [], "dense": ts}


# ----------------------------------------------------------------------

def run_one_video(cfg: SamplingConfig, video_path: str,
                   duration: float, video_id: str,
                   models, target_fps: float = 2.0) -> Dict:
    hl, fd, fe, sc = models
    cache = CrossTaskCache() if cfg.enable_cross_cache else None
    n_calls: Dict[str, int] = {"highlight": 0, "face_det": 0,
                                "face_emb": 0, "scene": 0}
    n_hits:  Dict[str, int] = {"highlight": 0, "face_det": 0,
                                "face_emb": 0, "scene": 0}

    t0 = time.perf_counter()

    # Stage A: prefilter
    if cfg.enable_prefilter:
        pre = MetadataPrefilter().analyze(video_path, duration)
    else:
        pre = None

    # Stage B: sampling
    ts_dict = _sample_timestamps(video_path, duration, cfg, target_fps)

    # Apply prefilter mask
    def _filter(ts):
        if pre is None or pre.candidate_mask is None:
            return ts
        mask = pre.candidate_mask
        return [t for t in ts
                 if 0 <= int(t * 10) < len(mask) and mask[int(t * 10)]]
    sparse_ts = _filter(ts_dict["sparse"])
    dense_ts  = _filter(ts_dict["dense"])

    # ---- Stage 1: sparse decode + (light) compute ----
    n_decoded = 0
    decode_passes = 1 if cfg.enable_unified_scheduler else 4
    sparse_frames = []
    if sparse_ts:
        if cfg.enable_unified_scheduler:
            sparse_frames = _decode_at(video_path, video_id, sparse_ts)
            n_decoded += len(sparse_frames)
        else:
            for _ in range(4):
                sparse_frames = _decode_at(video_path, video_id, sparse_ts)
            n_decoded += len(sparse_frames) * 4

    # ---- Stage 2: dense decode + main compute ----
    if cfg.enable_unified_scheduler:
        dense_frames = _decode_at(video_path, video_id, dense_ts)
        n_decoded += len(dense_frames)
    else:
        for _ in range(4):
            dense_frames = _decode_at(video_path, video_id, dense_ts)
        n_decoded += len(dense_frames) * 4

    # Fill cache from sparse frames so dense frames can hit
    all_frames = list(sparse_frames) + list(dense_frames)
    if not all_frames:
        return {
            "video_id": video_id, "wall_ms": 0,
            "n_frames_decoded": 0, "n_calls": n_calls, "n_hits": n_hits,
            "config": cfg.to_dict(),
        }

    fids_all = [FrameIdentity(f.image) for f in all_frames] \
                if cfg.enable_cross_cache else None

    # Cached call
    def _cached_call(model_id: str, fn, items, fids):
        if cache is None or fids is None:
            n_calls[model_id] += len(items)
            return fn(items)
        out = [None] * len(items); miss = []
        for i, fid in enumerate(fids):
            v = cache.get(fid, model_id=model_id, mode="byte")
            if v is not None:
                out[i] = v; n_hits[model_id] += 1
            else:
                miss.append(i)
        if miss:
            n_calls[model_id] += len(miss)
            res = fn([items[i] for i in miss])
            for j, i in enumerate(miss):
                out[i] = res[j]
                cache.put(fids[i], model_id, res[j])
        return out

    # If we have sparse frames, run a light pass on them (highlight + face_det)
    # so the cache populates BEFORE the dense pass (this is where reuse pays off).
    if sparse_frames and cfg.enable_two_stage:
        sf_imgs = [f.image for f in sparse_frames]
        sf_fids = fids_all[:len(sparse_frames)] if fids_all is not None else None
        _cached_call("highlight", hl.score, sf_imgs, sf_fids)
        _cached_call("face_det", fd.detect, sf_imgs, sf_fids)
        _cached_call("scene",    sc.classify, sf_imgs, sf_fids)

    # Main dense pass
    df_imgs = [f.image for f in dense_frames]
    df_fids = fids_all[len(sparse_frames):] if fids_all is not None else None

    hl_scores = _cached_call("highlight", hl.score, df_imgs, df_fids)
    fd_out    = _cached_call("face_det", fd.detect, df_imgs, df_fids)
    sc_out    = _cached_call("scene",    sc.classify, df_imgs, df_fids)
    pos_idx = [i for i, (p, _) in enumerate(fd_out) if p]
    if pos_idx:
        emb_imgs = [df_imgs[i] for i in pos_idx]
        emb_fids = [df_fids[i] for i in pos_idx] if df_fids is not None else None
        _cached_call("face_emb", fe.embed, emb_imgs, emb_fids)

    if cfg.enable_segment_aggregator and hl_scores:
        SegmentAggregator(percentile=0.7, merge_gap_sec=1.5,
                           min_segment_sec=0.5, max_segments=5
                           ).aggregate(list(zip([f.timestamp for f in dense_frames],
                                                hl_scores)))

    dt = (time.perf_counter() - t0) * 1000.0
    return {
        "video_id": video_id,
        "wall_ms": dt,
        "n_frames_decoded": n_decoded,
        "n_calls": dict(n_calls),
        "n_hits":  dict(n_hits),
        "decode_passes": decode_passes,
        "config": cfg.to_dict(),
    }


# ----------------------------------------------------------------------

def run_suite_on_videos(cfg_suite: List[SamplingConfig],
                          videos: List[Dict], models,
                          warmup_runs: int = 5,
                          n_repeats: int = 3) -> List[Dict]:
    """Run each config N times, report median wall_ms (immune to warmup).

    `warmup_runs`: extra dry-run passes BEFORE measurement starts.
    `n_repeats`:   repeat each config N times per video, take median.
    """
    rows = []

    # Heavy warmup: cycle through each model on multiple videos so ONNX
    # sessions, CUDA contexts, and decoder caches are all hot.
    if videos and warmup_runs > 0:
        print(f"[warmup] {warmup_runs} dry-run pass(es) "
              f"on the first {min(3, len(videos))} videos ...")
        warmup_cfg = SamplingConfig(name="WARMUP")
        for w in range(warmup_runs):
            for v in videos[: min(3, len(videos))]:
                try:
                    run_one_video(warmup_cfg, v["path"], v["duration"],
                                    v["id"], models)
                except Exception:
                    pass
        print("[warmup] done")

    for cfg in cfg_suite:
        per_vid_med = []
        all_results = []
        for v in videos:
            walls = []
            r_last = None
            for _ in range(n_repeats):
                try:
                    r = run_one_video(cfg, v["path"], v["duration"], v["id"],
                                        models)
                    walls.append(r["wall_ms"])
                    r_last = r
                except Exception as e:
                    print(f"  [warn] {cfg.name}/{v['id']}: {e}")
            if r_last and walls:
                r_last["wall_ms"] = float(np.median(walls))
                all_results.append(r_last)
        valid = [r for r in all_results if r["n_frames_decoded"] > 0]
        if not valid:
            rows.append({"name": cfg.name, "error": "all videos failed"})
            continue
        med_wall = float(np.median([r["wall_ms"] for r in valid]))
        avg_dec  = float(np.mean([r["n_frames_decoded"] for r in valid]))
        sum_calls = {k: sum(r["n_calls"][k] for r in valid)
                     for k in ("highlight", "face_det", "face_emb", "scene")}
        sum_hits = {k: sum(r["n_hits"].get(k, 0) for r in valid)
                    for k in sum_calls}
        rows.append({
            "name": cfg.name,
            "med_wall_ms": med_wall,
            "avg_n_frames_decoded": avg_dec,
            "sum_calls": sum_calls,
            "sum_hits":  sum_hits,
            "total_calls": int(sum(sum_calls.values())),
            "total_hits":  int(sum(sum_hits.values())),
            "n_videos_ok": len(valid),
            "config": cfg.to_dict(),
        })
        print(f"  {cfg.name:<28} med_wall={med_wall:7.0f}ms  "
              f"frames={avg_dec:6.1f}  calls={sum(sum_calls.values()):4d}  "
              f"hits={sum(sum_hits.values()):4d}")
    return rows


def make_full_suite() -> List[SamplingConfig]:
    return [
        SamplingConfig(name="S0_full"),
        SamplingConfig(name="S1_no_prefilter",          enable_prefilter=False),
        SamplingConfig(name="S2_no_two_stage",          enable_two_stage=False),
        SamplingConfig(name="S3_no_unified_scheduler",  enable_unified_scheduler=False),
        SamplingConfig(name="S4_no_seg_aggregator",     enable_segment_aggregator=False),
        SamplingConfig(name="S5_no_adaptive_sampler",   enable_adaptive_sampler=False),
        SamplingConfig(name="S6_no_cross_cache",        enable_cross_cache=False),
        SamplingConfig(name="S7_no_qpp",                enable_qpp_routing=False),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--n-videos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeat each config N times, take median wall_ms")
    ap.add_argument("--out-dir", default="experiments/results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    vdir = Path(args.videos_dir)
    all_mp4 = sorted(vdir.glob("video*.mp4"))
    if not all_mp4:
        print("[error] no videos"); sys.exit(1)
    candidates = rng.sample(all_mp4, min(len(all_mp4), args.n_videos * 3))
    videos = []
    for p in candidates:
        if len(videos) >= args.n_videos: break
        info = probe_video(str(p))
        if 2.0 < info.get("duration", 0) <= args.max_duration:
            videos.append({"id": p.stem, "path": str(p),
                            "duration": info["duration"]})
    print(f"[setup] {len(videos)} videos")

    print("[load] models ...")
    models = (
        MomentDETRHighlightModel(),
        InsightFaceDetector(),
        InsightFaceEmbedder(),
        MobileNetV3SceneClassifier(),
    )

    print("\n=== SAMPLING PIPELINE ABLATION ===")
    rows = run_suite_on_videos(make_full_suite(), videos, models,
                                 warmup_runs=args.warmup,
                                 n_repeats=args.repeats)

    full = next((r for r in rows if r["name"] == "S0_full"), rows[0])
    print(f"\n{'name':<28} {'wall_ms':>9} {'frames':>7} {'Δwall%':>8} "
          f"{'calls':>6} {'hits':>6} {'Δcalls':>8} {'Δhits':>8}")
    print("-" * 90)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<28} ERROR"); continue
        d_wall = (r["med_wall_ms"] - full["med_wall_ms"]) / max(full["med_wall_ms"], 1) * 100
        d_calls = (r["total_calls"] - full["total_calls"]) / max(full["total_calls"], 1) * 100
        d_hits = (r["total_hits"] - full["total_hits"]) / max(full["total_hits"], 1) * 100 \
                 if full["total_hits"] > 0 else 0
        print(f"{r['name']:<28} {r['med_wall_ms']:>8.0f} "
              f"{r['avg_n_frames_decoded']:>7.1f} "
              f"{d_wall:>+7.1f}% {r['total_calls']:>6} {r['total_hits']:>6} "
              f"{d_calls:>+7.1f}% {d_hits:>+7.1f}%")

    Path(out_dir / "sampling_modules.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    csv_path = out_dir / "sampling_modules.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "med_wall_ms", "avg_frames_decoded",
                    "total_calls", "total_hits",
                    "calls_highlight", "calls_face_det",
                    "calls_face_emb", "calls_scene"])
        for r in rows:
            if "error" in r: continue
            sc = r["sum_calls"]
            w.writerow([r["name"], r["med_wall_ms"],
                        r["avg_n_frames_decoded"],
                        r["total_calls"], r["total_hits"],
                        sc["highlight"], sc["face_det"],
                        sc["face_emb"], sc["scene"]])
    print(f"\n[saved] {csv_path}")


if __name__ == "__main__":
    main()
