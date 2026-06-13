"""Build a clearly-labelled mock MSR-VTT cache for CQIN/CSER smoke tests.

The generated embeddings come from ``tasks.mock_models.MockCLIPModel``. They
validate data plumbing and routing code only; they are not paper-ready metrics.
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
from tasks.mock_models import MockCLIPModel


def _natural_video_key(path: Path) -> tuple:
    stem = path.stem
    suffix = stem.removeprefix("video")
    return (0, int(suffix)) if suffix.isdigit() else (1, stem)


def _pad_embeddings(embs: list[np.ndarray], n: int, dim: int) -> np.ndarray:
    if not embs:
        return np.zeros((n, dim), dtype=np.float32)
    out = list(embs[:n])
    while len(out) < n:
        out.append(out[-1].copy())
    return np.stack(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--video-list", default=None,
                    help="optional standard split list, one video id per line")
    ap.add_argument("--out-dir", default="data/msrvtt_mock_smoke")
    ap.add_argument("--n-videos", type=int, default=80)
    ap.add_argument("--captions-per-video", type=int, default=2)
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--hw", type=int, default=64)
    args = ap.parse_args()

    annotation = Path(args.annotation)
    videos_dir = Path(args.videos_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads(annotation.read_text(encoding="utf-8"))
    annotated_ids = {str(x["id"]) for x in data["images"]}
    captions: dict[str, list[str]] = defaultdict(list)
    for row in data["annotations"]:
        captions[str(row["image_id"])].append(str(row["caption"]))

    paths = {p.stem: p for p in videos_dir.glob("*.mp4") if p.stem in annotated_ids}
    if args.video_list:
        listed_ids = [
            x.strip() for x in Path(args.video_list).read_text().splitlines()
            if x.strip()
        ]
        available = [paths[video_id] for video_id in listed_ids if video_id in paths]
    else:
        available = sorted(paths.values(), key=_natural_video_key)
    selected = available[:args.n_videos]
    if len(selected) < args.n_videos:
        raise RuntimeError(
            f"requested {args.n_videos} videos, found only {len(selected)} standard videos"
        )

    clip = MockCLIPModel(dim=128)
    video_ids = []
    video_embs = []
    protos = []
    query_rows = []
    for i, path in enumerate(selected):
        video_id = path.stem
        frames = _decode_frames(str(path), args.n_frames, args.hw)
        frame_embs = _pad_embeddings(
            clip.encode_frames(list(frames)), args.n_frames, clip.dim
        )
        mean = frame_embs.mean(axis=0)
        mean /= np.linalg.norm(mean) + 1e-9
        video_ids.append(video_id)
        video_embs.append(mean.astype(np.float32))
        protos.append(frame_embs)
        for sentence in captions[video_id][:args.captions_per_video]:
            query_rows.append({"sentence": sentence, "video_id": video_id})
        if (i + 1) % 20 == 0 or i + 1 == len(selected):
            print(f"[decode] {i + 1}/{len(selected)}")

    csv_path = out / "msrvtt_mock_smoke.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "video_id"])
        writer.writeheader()
        writer.writerows(query_rows)

    cache_path = out / "msrvtt_mock_smoke_cache.npz"
    np.savez_compressed(
        cache_path,
        video_ids=np.asarray(video_ids),
        video_embs=np.stack(video_embs),
        protos=np.stack(protos),
        proto_counts=np.full(len(video_ids), args.n_frames, dtype=np.int32),
    )
    texts = [x["sentence"] for x in query_rows]
    text_embs = np.stack(clip.encode_text(texts)).astype(np.float32)
    text_path = out / "msrvtt_mock_smoke.text_embs.npy"
    np.save(text_path, text_embs)

    gallery_dir = out / "cser_gallery"
    gallery_dir.mkdir(exist_ok=True)
    gallery_manifest = gallery_dir / "manifest.json"
    gallery_manifest.write_text(json.dumps([
        {"id": path.stem, "path": str(path)} for path in selected
    ], indent=2), encoding="utf-8")

    manifest = {
        "kind": "mock_smoke_only",
        "paper_ready": False,
        "annotation": str(annotation),
        "videos_dir": str(videos_dir),
        "video_list": args.video_list,
        "n_available_standard_videos_at_build": len(available),
        "n_selected_videos": len(video_ids),
        "n_queries": len(query_rows),
        "cache": str(cache_path),
        "csv": str(csv_path),
        "text_embs": str(text_path),
        "cser_gallery": str(gallery_dir),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
