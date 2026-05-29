"""End-to-end benchmark for LiteVTR++ v2 (black-box model preserving).

Runs three flavors and produces a comparison table:

  - V1_online    — current framework (on-demand CLIP per query)
  - V2_online    — v2 components but NO offline index (sanity baseline)
  - V2_offline   — v2 with pre-built OfflineIndex (expected 10–50x)

Usage:
    python demo/run_benchmark_v2.py --videos demo/sample_videos \
                                     [--real-models]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (
    LiteVTRFrameworkV2, OfflineIndex, OfflineIndexBuilder,
    MetadataPrefilter, QueryPlanner, QueryPlannerConfig,
    CrossTaskCache, HybridSampler, MVBasedSampler, UniformSampler,
    QFrameSampler, QFrameConfig,
)
from tasks import (
    MockCLIPModel, MockHighlightModel, MockFaceDetector, MockFaceEmbedder,
    MockSceneClassifier, real_models,
)


DEFAULT_QUERIES = [
    "a person walking outdoors",
    "people talking indoors",
    "a close-up of a face",
    "a bright outdoor scene",
    "a still indoor room",
]


# ----------------------------------------------------------------------

def build_models(use_real: bool):
    if use_real:
        try:
            clip = real_models.RealCLIPModel()
            highlight = real_models.MomentDETRHighlightModel()
            face_det = real_models.InsightFaceDetector()
            face_emb = real_models.InsightFaceEmbedder()
            scene = real_models.MobileNetV3SceneClassifier()
            print("[models] using real backbones")
            return clip, highlight, face_det, face_emb, scene
        except Exception as e:
            print(f"[models] real-model failure ({e}); falling back to mocks")
    clip = MockCLIPModel(dim=128)
    return (clip, MockHighlightModel(), MockFaceDetector(),
            MockFaceEmbedder(dim=64), MockSceneClassifier())


def load_videos(videos_dir: str) -> List[Dict]:
    vdir = Path(videos_dir)
    manifest = vdir / "manifest.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        from core.decoder import probe_video
        entries = []
        for p in sorted(vdir.glob("*.mp4")):
            info = probe_video(str(p))
            entries.append({
                "id": p.stem, "path": str(p),
                "duration": info["duration"] or 30.0,
            })
    for e in entries:
        e["duration"] = float(e.get("duration", 30.0))
    return entries


# ----------------------------------------------------------------------
#  Encoders: tiny adapters so our components don't import real_models
# ----------------------------------------------------------------------

def make_clip_adapters(clip):
    def t_enc(q: str) -> np.ndarray:
        out = clip.encode_text([q])
        return np.asarray(out[0], dtype=np.float32)

    def i_enc(frames: List[np.ndarray]) -> np.ndarray:
        out = clip.encode_frames(frames)
        return np.asarray(out, dtype=np.float32)
    return t_enc, i_enc


# ----------------------------------------------------------------------
#  V2 offline run
# ----------------------------------------------------------------------

def run_v2_offline(videos, queries, clip, highlight,
                    face_det, face_emb, scene, out_path: Path):
    t_enc, i_enc = make_clip_adapters(clip)

    print("\n[V2/offline] building offline index ...")
    t0 = time.perf_counter()
    builder = OfflineIndexBuilder(
        image_encoder=i_enc,
        face_detector=face_det.detect,
        scene_classifier=scene.classify,
        sampler=HybridSampler(samplers=[
            MVBasedSampler(motion_tau=1.5, max_samples=16),
            UniformSampler(fps=0.5, max_samples=16),
        ], dedup_gap_sec=0.5, max_samples=20),
        k_values=(2, 4, 6),
        max_keyframes=20,
    )
    index = builder.build_gallery(
        [{"id": v["id"], "path": v["path"], "duration": v["duration"]}
         for v in videos],
        save_path=str(out_path),
        progress=True,
    )
    dt_index = (time.perf_counter() - t0) * 1000.0
    print(f"[V2/offline] index built in {dt_index:.0f} ms; "
          f"{index.summary()}")

    # Attach a QFrame sampler that piggybacks on the same clip
    hybrid = HybridSampler(samplers=[
        MVBasedSampler(motion_tau=1.5, max_samples=20),
        QFrameSampler(image_encoder=i_enc, text_encoder=t_enc,
                       config=QFrameConfig(probe_fps=1.0, top_k=8),
                       max_samples=12),
        UniformSampler(fps=0.5, max_samples=12),
    ], dedup_gap_sec=0.3, max_samples=30)

    fw = LiteVTRFrameworkV2(
        offline_index=index,
        huawei_clip_text_encode=t_enc,
        huawei_clip_image_encode=i_enc,
        highlight_model=highlight,
        face_detector=face_det, face_embedder=face_emb, scene_classifier=scene,
        prefilter=MetadataPrefilter(),
        query_planner=QueryPlanner(QueryPlannerConfig(
            easy_margin=0.05, hard_margin=0.015)),
        cross_cache=CrossTaskCache(max_size_per_model=4096),
        hybrid_sampler=hybrid,
    )

    videos_meta = {v["id"]: v for v in videos}
    rows = []
    for q in queries:
        r = fw.query(q, videos_meta=videos_meta, top_k=3)
        top_ids = [vid for vid, _ in r["top_k"]]
        rows.append({
            "query": q,
            "top_k": top_ids,
            "margin": r["plan"].margin,
            "difficulty": r["plan"].difficulty.value,
            **{k: v for k, v in r["stats"].items()
               if k in ("wall_ms", "encode_text_ms", "index_search_ms",
                        "stage2_model_ms", "n_frames_decoded")},
        })
        print(f"  [{q[:32]:<32}] wall={r['stats']['wall_ms']:.1f}ms  "
              f"plan={r['plan'].difficulty.value:<6}  "
              f"margin={r['plan'].margin:.3f}  top1={top_ids[0] if top_ids else '-'}")

    return {"rows": rows, "index_build_ms": dt_index,
            "index_summary": index.summary()}


# ----------------------------------------------------------------------
#  V1 online (baseline comparison)
# ----------------------------------------------------------------------

def run_v1_online(videos, queries, clip):
    """Simulate the v1 cost: encode every video's frames on each query."""
    t_enc, i_enc = make_clip_adapters(clip)
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        q_emb = t_enc(q)
        # A coarse simulate: sample 8 frames per video, encode each.
        per_query_total = 0
        for v in videos:
            try:
                import cv2
                cap = cv2.VideoCapture(v["path"])
                dur = v["duration"]
                imgs = []
                for tt in np.linspace(0.1, max(dur - 0.1, 1.0), 8):
                    cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000.0)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        imgs.append(frame[..., ::-1])  # BGR->RGB
                cap.release()
                if imgs:
                    _ = i_enc(imgs)
                    per_query_total += len(imgs)
            except Exception:
                pass
        dt = (time.perf_counter() - t0) * 1000.0
        rows.append({"query": q, "wall_ms": dt,
                     "n_frames_encoded": per_query_total})
        print(f"  [{q[:32]:<32}] wall={dt:.1f}ms  "
              f"frames={per_query_total}")
    return {"rows": rows}


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--queries", nargs="+", default=None)
    ap.add_argument("--real-models", action="store_true")
    ap.add_argument("--out", default="BENCHMARK_V2.json")
    ap.add_argument("--index-path", default="./offline_index.pkl")
    args = ap.parse_args()

    videos = load_videos(args.videos)
    if not videos:
        print(f"[error] no videos under {args.videos}")
        sys.exit(1)
    queries = args.queries or DEFAULT_QUERIES

    clip, highlight, face_det, face_emb, scene = build_models(args.real_models)

    print(f"\n=== Running V1_online baseline ({len(queries)} queries, "
          f"{len(videos)} videos) ===")
    v1 = run_v1_online(videos, queries, clip)

    print(f"\n=== Running V2_offline (with offline index) ===")
    v2 = run_v2_offline(videos, queries, clip, highlight,
                         face_det, face_emb, scene,
                         out_path=Path(args.index_path))

    # ------------------------------------------------------------------ summary
    v1_avg = np.mean([r["wall_ms"] for r in v1["rows"]])
    v2_avg = np.mean([r["wall_ms"] for r in v2["rows"]])
    speedup = v1_avg / max(v2_avg, 1e-6)

    print("\n" + "=" * 72)
    print(f"V1 online avg per query : {v1_avg:7.1f} ms")
    print(f"V2 offline avg per query: {v2_avg:7.1f} ms")
    print(f"Speedup                 : {speedup:5.1f}x")
    print(f"Index build             : {v2['index_build_ms']:7.1f} ms (one-off)")
    print("=" * 72)

    Path(args.out).write_text(json.dumps({
        "v1": v1, "v2": v2,
        "speedup": float(speedup),
        "v1_avg_ms": float(v1_avg),
        "v2_avg_ms": float(v2_avg),
    }, indent=2), encoding="utf-8")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
