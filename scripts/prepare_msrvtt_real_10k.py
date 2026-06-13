"""Prepare MSR-VTT 10K metadata for CSER-only runs.

This script is intentionally lighter than ``prepare_msrvtt_real_1k.py``.  CSER
can build and reuse its own expert cache through ``--gallery-cache``, so this
preparation step only maps the raw MSR-VTT annotation and video directory into:

* ``msrvtt10k_queries.csv`` with ``sentence,video_id`` columns.
* ``cser_gallery/manifest.json`` with absolute video paths.
* ``manifest.json`` describing counts and any annotation/video mismatches.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _video_sort_key(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else float("inf"), path.stem)


def _load_annotation(annotation: Path) -> tuple[list[str], dict[str, list[str]]]:
    data = json.loads(annotation.read_text(encoding="utf-8"))
    image_ids = [str(row["id"]) for row in data.get("images", [])]
    captions: dict[str, list[str]] = defaultdict(list)
    for row in data.get("annotations", []):
        captions[str(row["image_id"])].append(str(row["caption"]))
    return image_ids, captions


def _available_videos(videos_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path.resolve()
        for path in sorted(videos_dir.glob("*.mp4"), key=_video_sort_key)
    }


def _selected_ids(annotation_ids: Iterable[str],
                  videos_by_id: dict[str, Path],
                  captions: dict[str, list[str]],
                  max_videos: int | None) -> list[str]:
    out = []
    seen = set()
    for video_id in annotation_ids:
        if video_id in seen:
            continue
        seen.add(video_id)
        if video_id not in videos_by_id or not captions.get(video_id):
            continue
        out.append(video_id)
        if max_videos is not None and len(out) >= max_videos:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--out-dir", default="data/msrvtt_real_10k")
    ap.add_argument("--captions-per-video", type=int, default=1)
    ap.add_argument("--max-videos", type=int, default=None,
                    help="optional smoke/debug limit; omit for full 10K")
    args = ap.parse_args()

    if args.captions_per_video <= 0:
        ap.error("--captions-per-video must be positive")
    if args.max_videos is not None and args.max_videos <= 0:
        ap.error("--max-videos must be positive when provided")

    annotation = Path(args.annotation).resolve()
    videos_dir = Path(args.videos_dir).resolve()
    out = Path(args.out_dir).resolve()
    gallery_dir = out / "cser_gallery"
    out.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    annotation_ids, captions = _load_annotation(annotation)
    videos_by_id = _available_videos(videos_dir)
    selected = _selected_ids(
        annotation_ids, videos_by_id, captions, args.max_videos)

    missing_video_ids = [
        video_id for video_id in annotation_ids
        if video_id not in videos_by_id
    ]
    no_caption_ids = [
        video_id for video_id in annotation_ids
        if video_id in videos_by_id and not captions.get(video_id)
    ]

    query_rows: list[dict[str, str]] = []
    for video_id in selected:
        selected_captions = captions[video_id][: args.captions_per_video]
        for sentence in selected_captions:
            query_rows.append({"sentence": sentence, "video_id": video_id})

    csv_path = out / "msrvtt10k_queries.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "video_id"])
        writer.writeheader()
        writer.writerows(query_rows)

    gallery_manifest = [
        {"id": video_id, "path": str(videos_by_id[video_id])}
        for video_id in selected
    ]
    (gallery_dir / "manifest.json").write_text(
        json.dumps(gallery_manifest, indent=2), encoding="utf-8")

    manifest = {
        "kind": "real_msrvtt_10k_cser_only",
        "annotation": str(annotation),
        "videos_dir": str(videos_dir),
        "csv": str(csv_path),
        "cser_gallery": str(gallery_dir),
        "captions_per_video": int(args.captions_per_video),
        "max_videos": args.max_videos,
        "n_annotation_videos": len(annotation_ids),
        "n_video_files": len(videos_by_id),
        "n_selected_videos": len(selected),
        "n_queries": len(query_rows),
        "missing_video_ids": missing_video_ids,
        "no_caption_video_ids": no_caption_ids,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[saved] {out}", flush=True)
    print(f"[summary] selected_videos={len(selected)} queries={len(query_rows)} "
          f"missing_videos={len(missing_video_ids)} no_caption={len(no_caption_ids)}",
          flush=True)


if __name__ == "__main__":
    main()
