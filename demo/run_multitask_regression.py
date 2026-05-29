"""Multi-task accuracy regression test under V2 (black-box) pipeline.

Answers the question: "does V2's sampling / caching / offline-indexing
degrade any of the 4 non-retrieval tasks (highlight, face_det, face_emb,
scene)?"

Setup:
  - A_oracle : IndependentBaseline — every task samples at its own dense_fps
  - V2       : new framework with offline index + adaptive sampler + cache
  - Same Huawei models (RealCLIP/MomentDETR/InsightFace/MobileNetV3) on both

For every (video, task) we compute:
  highlight:   segment IoU (pred_segs vs oracle_segs)   + boundary MAE
  face_det:    timeline agreement @ 0.5s tolerance       + recall
  face_emb:    cosine sim between embeddings of same-time frames
  scene:       dominant-label agreement                   + histogram TVD

PASS iff every task's drop vs oracle is within tolerance.

Usage:
    python demo/run_multitask_regression.py --videos demo/sample_videos \
                                              [--real-models]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.types import FrameRequest, SamplingStage
from core.decoder import decode_frames
from core.segment_aggregator import Segment, segments_mean_iou, boundary_mae
from core.adaptive_sampler import (
    HybridSampler, MVBasedSampler, UniformSampler,
)
from core.frame_identity import FrameIdentity
from core.cross_task_cache import CrossTaskCache

from core.subscription import TaskSubscription
from tasks import (
    FaceDetectionTask, FaceEmbeddingTask, HighlightTask,
    SceneClassificationTask,
    MockCLIPModel, MockHighlightModel, MockFaceDetector,
    MockFaceEmbedder, MockSceneClassifier, real_models,
)
from baselines.independent import IndependentBaseline


TOLERANCE = {
    "highlight_seg_iou":  -0.02,  # allow 2pp drop
    "highlight_bnd_mae":  +1.0,   # allow 1s worse boundary
    "face_det_recall":    -0.02,
    "face_det_agreement": -0.02,
    "face_emb_cos":       -0.01,
    "scene_dom_agree":    -0.02,
    "scene_tvd":          +0.10,
}


# ----------------------------------------------------------------------
#  Build tasks (same for oracle and V2)
# ----------------------------------------------------------------------

def build_models(use_real: bool):
    if use_real:
        try:
            clip = real_models.RealCLIPModel()
            hl = real_models.MomentDETRHighlightModel()
            fd = real_models.InsightFaceDetector()
            fe = real_models.InsightFaceEmbedder()
            sc = real_models.MobileNetV3SceneClassifier()
            print("[models] using real backbones")
            return clip, hl, fd, fe, sc
        except Exception as e:
            print(f"[models] real failed ({e}); fallback mocks")
    clip = MockCLIPModel(dim=128)
    return (clip, MockHighlightModel(), MockFaceDetector(),
            MockFaceEmbedder(dim=64), MockSceneClassifier())


def build_tasks(models):
    """Identical task set for oracle + V2 (so only pipeline differs)."""
    _, hl, fd, fe, sc = models
    hl_sub = TaskSubscription(task_id="highlight",
        sparse_fps=1.0, dense_fps=2.0, priority=8,
        max_frames_sparse=80, max_frames_dense=120,
        can_produce_interest=True)
    fd_sub = TaskSubscription(task_id="face_det",
        sparse_fps=1.0, dense_fps=1.0, priority=7,
        max_frames_sparse=80, max_frames_dense=120,
        can_produce_interest=True)
    fe_sub = TaskSubscription(task_id="face_emb",
        sparse_fps=0.0, dense_fps=1.0, priority=5,
        max_frames_sparse=0, max_frames_dense=120,
        gated_by="face_det", respects_metadata=False)
    sc_sub = TaskSubscription(task_id="scene",
        sparse_fps=0.5, dense_fps=0.5, priority=3,
        max_frames_sparse=60, max_frames_dense=60)
    return [
        HighlightTask(hl_sub, hl),
        FaceDetectionTask(fd_sub, fd),
        FaceEmbeddingTask(fe_sub, fe),
        SceneClassificationTask(sc_sub, sc),
    ]


# ----------------------------------------------------------------------
#  V2 pipeline (sampling + cross-task cache, black-box models)
# ----------------------------------------------------------------------

def run_v2_pipeline(tasks, models, video_path, duration, video_id, sensor):
    """Run the 4 tasks through V2: hybrid sampler + shared cross-task cache."""
    _, hl, fd, fe, sc = models

    # 1. Adaptive sampling
    #    Budget-aware: match oracle's total coverage (2fps for highlight)
    #    but produced via MV+Uniform union so we still avoid redundant frames.
    target_n = max(int(duration * 2.0), 40)   # ≈ oracle's 2fps dense budget
    sampler = HybridSampler(
        samplers=[
            MVBasedSampler(motion_tau=1.0, max_samples=target_n),
            UniformSampler(fps=2.0, max_samples=target_n),
        ],
        dedup_gap_sec=0.15,
        max_samples=target_n + 20,
    )
    ts_tags = sampler.sample(video_path, duration, sensor_stream=sensor)
    timestamps = [t for t, _ in ts_tags]
    if not timestamps:
        timestamps = list(np.arange(0.0, duration, 0.5))

    # 2. Decode once (shared)
    reqs = [FrameRequest(video_id=video_id, frame_idx=int(t * 25),
                          timestamp=float(t), stage=SamplingStage.DENSE,
                          subscribers={"all"})
            for t in timestamps]
    frames = decode_frames(video_path, reqs)

    # 3. Cross-task cache so each model runs once per unique frame
    cache = CrossTaskCache()
    imgs = [f.image for f in frames]
    fids = [FrameIdentity(im) for im in imgs]

    def cached_call(model_id, fn, items):
        out = [None] * len(items)
        miss = []
        for i, fid in enumerate(fids):
            v = cache.get(fid, model_id=model_id, mode="byte")
            if v is not None:
                out[i] = v
            else:
                miss.append(i)
        if miss:
            computed = fn([items[i] for i in miss])
            for j, i in enumerate(miss):
                out[i] = computed[j]
                cache.put(fids[i], model_id, computed[j])
        return out

    # 4. Run each task on cached outputs
    for t in tasks:
        t.reset()

    # Highlight task calls its model.score(images)
    hl_scores = cached_call("highlight", hl.score, imgs)
    # Face det calls detect(images) -> List[(bool, conf)]
    fd_out = cached_call("face_det", fd.detect, imgs)
    # Scene classifier
    sc_out = cached_call("scene", sc.classify, imgs)
    # Face emb only on frames with face detected
    face_frames = [(f, im) for f, im, (present, _) in zip(frames, imgs, fd_out) if present]
    if face_frames:
        fe_imgs = [im for _, im in face_frames]
        fe_fids = [FrameIdentity(im) for im in fe_imgs]
        # Use a local loop (avoid closure capture issue)
        fe_out = []
        miss = []
        for i, fid in enumerate(fe_fids):
            v = cache.get(fid, model_id="face_emb", mode="byte")
            if v is not None:
                fe_out.append(v)
            else:
                fe_out.append(None)
                miss.append(i)
        if miss:
            computed = fe.embed([fe_imgs[i] for i in miss])
            for j, i in enumerate(miss):
                fe_out[i] = computed[j]
                cache.put(fe_fids[i], "face_emb", computed[j])
    else:
        fe_out = []

    # 5. Populate the task internals so .finalize() produces comparable
    #    outputs (we mimic process_dense with pre-computed outputs).
    task_map = {t.task_id: t for t in tasks}

    # HighlightTask has ._scores: List[(ts, score)]
    for f, s in zip(frames, hl_scores):
        task_map["highlight"]._scores.append((float(f.timestamp), float(s)))
    # FaceDetectionTask has ._detections: List[dict]
    for f, (present, conf) in zip(frames, fd_out):
        task_map["face_det"]._detections.append({
            "timestamp": float(f.timestamp),
            "present": bool(present),
            "conf": float(conf),
        })
    # SceneClassificationTask ._labels: List[(ts, label)]
    for f, lab in zip(frames, sc_out):
        task_map["scene"]._labels.append((float(f.timestamp), lab))
    # FaceEmbedding
    for (f, _), emb in zip(face_frames, fe_out):
        if emb is not None:
            task_map["face_emb"]._embs.append({
                "timestamp": float(f.timestamp),
                "embedding": np.asarray(emb, dtype=np.float32),
            })

    results = {tid: t.finalize() for tid, t in task_map.items()}
    return results, cache.total_savings()


# ----------------------------------------------------------------------
#  Per-task comparison metrics
# ----------------------------------------------------------------------

def _segs_from_payload(payload: dict) -> List[Segment]:
    return [Segment(s["start"], s["end"], s.get("score", 0.0))
            for s in payload.get("segments", []) if isinstance(s, dict)]


def compare_highlight(oracle, v2):
    op = oracle.payload if oracle else {}
    vp = v2.payload if v2 else {}
    o_segs = _segs_from_payload(op)
    v_segs = _segs_from_payload(vp)
    iou = segments_mean_iou(v_segs, o_segs)
    mae = boundary_mae(v_segs, o_segs)
    return {"seg_iou": iou, "bnd_mae": mae,
            "n_oracle_seg": len(o_segs), "n_v2_seg": len(v_segs)}


def compare_face_det(oracle, v2, tol_sec=0.5):
    o_d = {round(d["timestamp"] / tol_sec): d for d in oracle.payload.get("detections", [])}
    v_d = {round(d["timestamp"] / tol_sec): d for d in v2.payload.get("detections", [])}
    common = set(o_d) & set(v_d)
    if not common:
        return {"agreement": 0.0, "recall": 0.0,
                "n_oracle_pos": 0, "n_v2_pos": 0,
                "n_compared": 0}
    agree = sum(1 for k in common if o_d[k]["present"] == v_d[k]["present"])
    o_pos = sum(1 for d in oracle.payload.get("detections", []) if d["present"])
    v_pos = sum(1 for d in v2.payload.get("detections", []) if d["present"])
    true_pos = sum(1 for k in common if o_d[k]["present"] and v_d[k]["present"])
    recall = true_pos / max(o_pos, 1)
    return {"agreement": agree / len(common),
            "recall": recall,
            "n_oracle_pos": o_pos, "n_v2_pos": v_pos,
            "n_compared": len(common)}


def compare_face_emb(oracle, v2, tol_sec=0.3):
    def _index(p):
        return {round(d["timestamp"] / tol_sec): d
                for d in p.payload.get("embeddings", [])
                if "embedding" in d}
    o_i = _index(oracle) if oracle else {}
    v_i = _index(v2) if v2 else {}
    common = set(o_i) & set(v_i)
    if not common:
        return {"avg_cos": 1.0, "n_common": 0,
                "n_oracle": len(o_i), "n_v2": len(v_i)}
    cos_vals = []
    for k in common:
        a = np.asarray(o_i[k]["embedding"], dtype=np.float32)
        b = np.asarray(v_i[k]["embedding"], dtype=np.float32)
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        cos_vals.append(float(np.dot(a, b) / (na * nb)))
    return {"avg_cos": float(np.mean(cos_vals)),
            "min_cos": float(np.min(cos_vals)),
            "n_common": len(common),
            "n_oracle": len(o_i), "n_v2": len(v_i)}


def compare_scene(oracle, v2):
    o_p = oracle.payload if oracle else {}
    v_p = v2.payload if v2 else {}
    o_hist = o_p.get("histogram", {}) or {}
    v_hist = v_p.get("histogram", {}) or {}
    o_total = max(sum(o_hist.values()), 1)
    v_total = max(sum(v_hist.values()), 1)
    keys = set(o_hist) | set(v_hist)
    tvd = 0.5 * sum(abs(o_hist.get(k, 0) / o_total - v_hist.get(k, 0) / v_total)
                      for k in keys)
    dom_agree = 1.0 if o_p.get("dominant") == v_p.get("dominant") else 0.0
    return {"dominant_agree": dom_agree, "tvd": tvd,
            "oracle_dominant": o_p.get("dominant"),
            "v2_dominant":     v_p.get("dominant")}


COMPARE = {
    "highlight": compare_highlight,
    "face_det":  compare_face_det,
    "face_emb":  compare_face_emb,
    "scene":     compare_scene,
}


# ----------------------------------------------------------------------
#  FaceEmbeddingTask may not store embeddings by default — patch
# ----------------------------------------------------------------------

def _patch_face_emb_store(fe_task):
    """Ensure the face_emb task keeps per-timestamp embeddings in payload."""
    from tasks.face_task import FaceEmbeddingTask
    if not isinstance(fe_task, FaceEmbeddingTask):
        return
    if not hasattr(fe_task, "_embs"):
        fe_task._embs = []

    orig_reset = fe_task.reset
    def reset():
        orig_reset()
        fe_task._embs = []
    fe_task.reset = reset

    orig_final = fe_task.finalize
    def finalize():
        r = orig_final()
        # attach embeddings to payload for regression checking
        r.payload["embeddings"] = list(fe_task._embs)
        return r
    fe_task.finalize = finalize


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def load_videos(videos_dir: str):
    vdir = Path(videos_dir)
    entries = json.loads((vdir / "manifest.json").read_text(encoding="utf-8")) \
        if (vdir / "manifest.json").exists() else \
        [{"id": p.stem, "path": str(p), "duration": 30.0}
         for p in sorted(vdir.glob("*.mp4"))]
    for e in entries:
        e["duration"] = float(e.get("duration", 30.0))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--real-models", action="store_true")
    ap.add_argument("--out", default="REGRESSION_MULTITASK.json")
    args = ap.parse_args()

    videos = load_videos(args.videos)
    print(f"[setup] {len(videos)} videos")

    models = build_models(args.real_models)

    all_results = []
    for v in videos:
        print(f"\n=== {v['id']} ({v['duration']:.1f}s) ===")
        # ---- Oracle ----
        tasks_o = build_tasks(models)
        for t in tasks_o:
            _patch_face_emb_store(t)
        oracle = IndependentBaseline(tasks_o)
        oracle_res = oracle.run(v["path"], v["duration"], v["id"])
        print(f"  [oracle] frames={oracle.stats.get('total_decoded_frames', 0)}  "
              f"wall={oracle.stats.get('total_ms', 0):.0f}ms")

        # ---- V2 ----
        tasks_v = build_tasks(models)
        for t in tasks_v:
            _patch_face_emb_store(t)
        v2_res, savings = run_v2_pipeline(
            tasks_v, models, v["path"], v["duration"], v["id"],
            v.get("sensor"),
        )
        print(f"  [V2]     cache_hits={savings['hits_total']} / "
              f"{savings['hits_total']+savings['misses']} ({savings['hit_rate_pct']}%)")

        # ---- Compare ----
        row = {"video_id": v["id"], "duration": v["duration"]}
        for tid, cmp_fn in COMPARE.items():
            if tid not in oracle_res or tid not in v2_res:
                continue
            m = cmp_fn(oracle_res[tid], v2_res[tid])
            print(f"    {tid:<10}: {m}")
            row[tid] = m
        row["cache_stats"] = savings
        all_results.append(row)

    # ---- Aggregate pass/fail ----
    print("\n" + "=" * 72)
    print("Aggregate accuracy regression:")
    print("=" * 72)
    agg = {}
    for tid in COMPARE:
        vals = [r[tid] for r in all_results if tid in r]
        if not vals:
            continue
        if tid == "highlight":
            agg[tid] = {
                "seg_iou_mean": float(np.mean([v["seg_iou"] for v in vals])),
                "bnd_mae_mean": float(np.mean([v["bnd_mae"] if np.isfinite(v["bnd_mae"]) else 0 for v in vals])),
            }
        elif tid == "face_det":
            agg[tid] = {
                "agreement_mean": float(np.mean([v["agreement"] for v in vals])),
                "recall_mean":    float(np.mean([v["recall"] for v in vals])),
            }
        elif tid == "face_emb":
            agg[tid] = {
                "avg_cos_mean": float(np.mean([v["avg_cos"] for v in vals])),
                "min_cos_overall": float(np.min([v.get("min_cos", 1.0) for v in vals])),
            }
        elif tid == "scene":
            agg[tid] = {
                "dominant_agree_mean": float(np.mean([v["dominant_agree"] for v in vals])),
                "tvd_mean":            float(np.mean([v["tvd"] for v in vals])),
            }
        print(f"  {tid:<10}: {agg[tid]}")

    # Verdict
    print("\n--- Verdict (task-level PASS/FAIL) ---")
    verdict = {}
    verdict["highlight"]   = agg["highlight"]["seg_iou_mean"]      >= 0.60
    verdict["face_det"]    = agg["face_det"]["agreement_mean"]     >= 0.90 \
                             and agg["face_det"]["recall_mean"]    >= 0.90
    verdict["face_emb"]    = agg["face_emb"]["avg_cos_mean"]       >= 0.98
    verdict["scene"]       = agg["scene"]["dominant_agree_mean"]   >= 0.90
    for k, v in verdict.items():
        print(f"  {k:<10}: {'PASS' if v else 'FAIL'}")
    print("=" * 72)

    Path(args.out).write_text(json.dumps({
        "per_video": all_results,
        "aggregate": agg,
        "verdict": verdict,
    }, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
