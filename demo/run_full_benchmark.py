"""Run the complete multi-model benchmark: 5 strategies x N videos x 5 tasks.

Usage:
    # Mock models (fast, no weights required)
    python demo/run_full_benchmark.py --videos demo/sample_videos

    # Real models (MobileCLIP2 + MomentDETR + InsightFace + MobileNetV3)
    # Falls back to Mock for any component that fails to load.
    python demo/run_full_benchmark.py --videos demo/sample_videos --real-models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `core`, `tasks`, etc. import cleanly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.subscription import TaskSubscription
from tasks import (
    FaceDetectionTask,
    FaceEmbeddingTask,
    HighlightTask,
    RetrievalTask,
    SceneClassificationTask,
    MockCLIPModel,
    MockFaceDetector,
    MockFaceEmbedder,
    MockHighlightModel,
    MockSceneClassifier,
    make_query_embeddings,
    real_models,
)
from benchmark.runner import BenchmarkRunner


DEFAULT_TEXT_QUERIES = [
    "a person walking outdoors",
    "people talking indoors",
    "a close-up of a face",
]


def _try_real(name: str, ctor, fallback):
    """Build a real model; on any failure, silently fall back to Mock."""
    try:
        obj = ctor()
        print(f"  [real] {name}: loaded")
        return obj
    except Exception as e:
        print(f"  [mock] {name}: {type(e).__name__}: {e}")
        return fallback()


def build_models(use_real: bool, text_queries=None) -> dict:
    """Return shared model instances (real if possible, else mock)."""
    text_queries = text_queries or DEFAULT_TEXT_QUERIES
    models: dict = {}

    if use_real:
        print("[models] real-model mode (auto-fallback to Mock on error)")
        models["clip"] = _try_real(
            "CLIP (MobileCLIP2-S0)",
            lambda: real_models.RealCLIPModel(),
            lambda: MockCLIPModel(dim=128),
        )
        if isinstance(models["clip"], real_models.RealCLIPModel):
            try:
                models["queries"] = models["clip"].encode_text(text_queries)
                print(f"  [real] queries: {len(text_queries)} texts encoded")
            except Exception as e:
                print(f"  [mock] queries fallback: {e}")
                models["clip"] = MockCLIPModel(dim=128)
                models["queries"] = make_query_embeddings(
                    n=len(text_queries), dim=128
                )
        else:
            models["queries"] = make_query_embeddings(
                n=len(text_queries), dim=128
            )

        models["highlight"] = _try_real(
            "Highlight (MomentDETR)",
            lambda: real_models.MomentDETRHighlightModel(),
            lambda: MockHighlightModel(),
        )
        models["face_det"] = _try_real(
            "FaceDet (InsightFace)",
            lambda: real_models.InsightFaceDetector(),
            lambda: MockFaceDetector(),
        )
        models["face_emb"] = _try_real(
            "FaceEmb (InsightFace)",
            lambda: real_models.InsightFaceEmbedder(),
            lambda: MockFaceEmbedder(dim=64),
        )
        models["scene"] = _try_real(
            "Scene (MobileNetV3)",
            lambda: real_models.MobileNetV3SceneClassifier(),
            lambda: MockSceneClassifier(),
        )
    else:
        print("[models] mock-model mode")
        models["clip"] = MockCLIPModel(dim=128)
        models["queries"] = make_query_embeddings(n=3, dim=128)
        models["highlight"] = MockHighlightModel()
        models["face_det"] = MockFaceDetector()
        models["face_emb"] = MockFaceEmbedder(dim=64)
        models["scene"] = MockSceneClassifier()

    return models


def make_build_tasks(models: dict):
    """Factory closure — fresh task instances for each strategy/video run."""
    def build_tasks():
        retrieval_sub = TaskSubscription(
            task_id="retrieval",
            sparse_fps=1.0, dense_fps=2.0,
            priority=10, max_frames_sparse=80, max_frames_dense=120,
            can_produce_interest=True,
        )
        highlight_sub = TaskSubscription(
            task_id="highlight",
            sparse_fps=1.0, dense_fps=2.0,
            priority=8, max_frames_sparse=80, max_frames_dense=120,
            can_produce_interest=True,
        )
        face_det_sub = TaskSubscription(
            task_id="face_det",
            sparse_fps=1.0, dense_fps=1.0,
            priority=7, max_frames_sparse=80, max_frames_dense=120,
            can_produce_interest=True,
        )
        face_emb_sub = TaskSubscription(
            task_id="face_emb",
            sparse_fps=0.0, dense_fps=1.0,
            priority=5, max_frames_sparse=0, max_frames_dense=120,
            gated_by="face_det",
            respects_metadata=False,
        )
        scene_sub = TaskSubscription(
            task_id="scene",
            sparse_fps=0.5, dense_fps=0.5,
            priority=3, max_frames_sparse=60, max_frames_dense=60,
        )
        return [
            RetrievalTask(retrieval_sub, models["clip"], models["queries"],
                           top_k=5),
            HighlightTask(highlight_sub, models["highlight"]),
            FaceDetectionTask(face_det_sub, models["face_det"]),
            FaceEmbeddingTask(face_emb_sub, models["face_emb"]),
            SceneClassificationTask(scene_sub, models["scene"]),
        ]
    return build_tasks


def load_videos(videos_dir: str):
    """Load manifest.json and attach sensor stream from meta files."""
    vdir = Path(videos_dir)
    manifest_path = vdir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    else:
        from core.decoder import probe_video
        entries = []
        for p in sorted(vdir.glob("*.mp4")):
            info = probe_video(str(p))
            entries.append({
                "id": p.stem,
                "path": str(p),
                "duration": info["duration"] or 30.0,
                "meta": None,
            })

    videos = []
    for e in entries:
        sensor = None
        meta_path = e.get("meta")
        if meta_path and Path(meta_path).exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            sensor = meta.get("sensor")
        videos.append({
            "id": e["id"],
            "path": e["path"],
            "duration": float(e["duration"]),
            "sensor": sensor,
        })
    return videos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True,
                        help="Directory containing videos and manifest.json")
    parser.add_argument("--output", default="BENCHMARK_REPORT.md")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--real-models", action="store_true",
                        help="Use real MobileCLIP2/MomentDETR/InsightFace/"
                             "MobileNetV3 weights (falls back to Mock per "
                             "component on load failure).")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Text queries for retrieval (real-model mode). "
                             "Default: 3 generic prompts.")
    args = parser.parse_args()

    videos = load_videos(args.videos)
    if not videos:
        print(f"[error] no videos found under {args.videos}")
        sys.exit(1)

    print(f"[setup] {len(videos)} videos, 5 strategies, 5 tasks")
    print("        strategies: A_independent, B_union_fps, C_framework, "
          "C1_no_prefilter, C2_no_two_stage")

    models = build_models(use_real=args.real_models, text_queries=args.queries)
    build_tasks = make_build_tasks(models)

    runner = BenchmarkRunner(
        videos=videos,
        tasks_factory=build_tasks,
        output_dir=args.output_dir,
        oracle_strategy="A_independent",
    )
    runner.run_all(
        report_path=args.output,
        raw_path="benchmark_raw.json",
    )


if __name__ == "__main__":
    main()
