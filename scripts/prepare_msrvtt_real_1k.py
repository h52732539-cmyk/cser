"""Build a resumable real-MobileCLIP MSR-VTT 1K cache for CQIN and CSER.

The standard MSR-VTT 1K split contains 1000 gallery videos.  This script
selects one caption per video, caches six MobileCLIP2-S0 frame embeddings per
video, writes CQIN's offline-index inputs, and writes a CSER gallery manifest.

Per-video frame embeddings are stored separately so an interrupted Slurm job
can resume without recomputing completed videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cser.data import _decode_frames
from tasks.real_models import RealCLIPModel


def _l2_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / (np.linalg.norm(array, axis=-1, keepdims=True) + 1e-9)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def _pad_embeddings(embs: list[np.ndarray], n: int, dim: int) -> np.ndarray:
    if not embs:
        return np.zeros((n, dim), dtype=np.float32)
    out = list(embs[:n])
    while len(out) < n:
        out.append(out[-1].copy())
    return _l2_rows(np.stack(out))


def _load_split(video_list: Path, videos_dir: Path, annotated_ids: set[str]) -> list[Path]:
    paths = {
        p.stem: p
        for p in videos_dir.glob("*.mp4")
        if p.stem in annotated_ids
    }
    listed_ids = [x.strip() for x in video_list.read_text().splitlines() if x.strip()]
    missing = [video_id for video_id in listed_ids if video_id not in paths]
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(f"{len(missing)} split videos are missing, first entries: {sample}")
    if len(listed_ids) != 1000:
        raise RuntimeError(f"expected the standard 1000-video split, found {len(listed_ids)} ids")
    return [paths[video_id] for video_id in listed_ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--video-list", required=True)
    ap.add_argument("--out-dir", default="data/msrvtt_real_1k")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--hw", type=int, default=224)
    ap.add_argument("--captions-per-video", type=int, default=1)
    ap.add_argument("--text-batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    annotation = Path(args.annotation).resolve()
    videos_dir = Path(args.videos_dir).resolve()
    video_list = Path(args.video_list).resolve()
    out = Path(args.out_dir).resolve()
    feature_dir = out / "frame_features"
    gallery_dir = out / "cser_gallery"
    out.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(exist_ok=True)
    gallery_dir.mkdir(exist_ok=True)

    data = json.loads(annotation.read_text(encoding="utf-8"))
    annotated_ids = {str(x["id"]) for x in data["images"]}
    captions: dict[str, list[str]] = defaultdict(list)
    for row in data["annotations"]:
        captions[str(row["image_id"])].append(str(row["caption"]))

    selected = _load_split(video_list, videos_dir, annotated_ids)
    query_rows: list[dict[str, str]] = []
    for path in selected:
        video_id = path.stem
        selected_captions = captions[video_id][: args.captions_per_video]
        if len(selected_captions) < args.captions_per_video:
            raise RuntimeError(f"video {video_id} has no usable caption")
        for sentence in selected_captions:
            query_rows.append({"sentence": sentence, "video_id": video_id})

    csv_path = out / "msrvtt_test_1k.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "video_id"])
        writer.writeheader()
        writer.writerows(query_rows)

    (gallery_dir / "manifest.json").write_text(
        json.dumps(
            [{"id": path.stem, "path": str(path)} for path in selected],
            indent=2,
        ),
        encoding="utf-8",
    )

    clip = RealCLIPModel()
    video_ids: list[str] = []
    video_embs: list[np.ndarray] = []
    protos: list[np.ndarray] = []
    for i, path in enumerate(selected):
        video_id = path.stem
        feature_path = feature_dir / f"{video_id}.npy"
        frame_embs: np.ndarray | None = None
        if feature_path.exists() and not args.force:
            cached = np.load(feature_path)
            if cached.shape == (args.n_frames, clip.dim):
                frame_embs = _l2_rows(cached)
        if frame_embs is None:
            frames = _decode_frames(str(path), args.n_frames, args.hw)
            frame_embs = _pad_embeddings(
                clip.encode_frames(list(frames)), args.n_frames, clip.dim
            )
            _atomic_save_npy(feature_path, frame_embs)

        mean = frame_embs.mean(axis=0)
        mean /= np.linalg.norm(mean) + 1e-9
        video_ids.append(video_id)
        video_embs.append(mean.astype(np.float32))
        protos.append(frame_embs.astype(np.float32))
        if (i + 1) % 20 == 0 or i + 1 == len(selected):
            print(f"[video-cache] {i + 1}/{len(selected)}", flush=True)

    cache_path = out / "msrvtt_cache.npz"
    _atomic_save_npz(
        cache_path,
        video_ids=np.asarray(video_ids),
        video_embs=np.stack(video_embs),
        protos=np.stack(protos),
        proto_counts=np.full(len(video_ids), args.n_frames, dtype=np.int32),
    )

    text_path = out / "msrvtt_test_1k.text_embs.npy"
    text_embs: np.ndarray | None = None
    if text_path.exists() and not args.force:
        cached = np.load(text_path)
        if cached.shape == (len(query_rows), clip.dim):
            text_embs = _l2_rows(cached)
    if text_embs is None:
        texts = [row["sentence"] for row in query_rows]
        batches = []
        for start in range(0, len(texts), args.text_batch_size):
            batch = texts[start : start + args.text_batch_size]
            batches.extend(clip.encode_text(batch))
            print(f"[text-cache] {min(start + len(batch), len(texts))}/{len(texts)}",
                  flush=True)
        text_embs = _l2_rows(np.stack(batches))
        _atomic_save_npy(text_path, text_embs)

    manifest = {
        "kind": "real_msrvtt_1k",
        "paper_ready": True,
        "annotation": str(annotation),
        "videos_dir": str(videos_dir),
        "video_list": str(video_list),
        "n_selected_videos": len(video_ids),
        "n_queries": len(query_rows),
        "captions_per_video": args.captions_per_video,
        "n_frames": args.n_frames,
        "cache": str(cache_path),
        "csv": str(csv_path),
        "text_embs": str(text_path),
        "cser_gallery": str(gallery_dir),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
