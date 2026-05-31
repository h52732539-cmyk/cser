"""Data layer for CSER — real-expert version, gallery-agnostic.

A gallery is now a set of *videos with frames* (the 5 expert models run over
frames). Two sources, one interface (:class:`Dataset`):

  * :func:`build_synthetic_dataset` — self-contained; generates videos as small
    frame stacks with controlled structure so the mock (or real) expert models
    produce varied, query-correlated signals. Runs with NO external files.
  * :func:`load_video_dataset` — real path: decode frames from a directory of
    videos (or a manifest), then the same pipeline applies.

Each query carries text + a CLIP text embedding (from the bundle's encoder) and
its ground-truth video id. Query priors (person/highlight/scene cues) are derived
in ``expert_features.build_query_priors``.

The heavy work — running the 5 models over every video — is done once by
``expert_features.extract_gallery_signals`` and cached in ``Dataset.gallery``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .expert_features import (ModelBundle, build_model_bundle,
                              extract_gallery_signals, GallerySignals,
                              build_query_priors, QueryExpertPriors)


@dataclass
class Dataset:
    gallery: GallerySignals               # cached per-video expert signals
    video_ids: List[str]
    query_texts: List[str]
    query_priors: List[QueryExpertPriors]
    gt_video_ids: List[str]
    bundle: ModelBundle

    @property
    def n_queries(self) -> int:
        return len(self.gt_video_ids)

    @property
    def gallery_size(self) -> int:
        return self.gallery.size

    def split(self, fracs=(0.60, 0.15, 0.25), seed: int = 42
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_queries)
        n_tr = int(self.n_queries * fracs[0])
        n_cal = int(self.n_queries * fracs[1])
        return perm[:n_tr], perm[n_tr:n_tr + n_cal], perm[n_tr + n_cal:]


# ----------------------------------------------------------------------
#  Synthetic gallery: videos as frame stacks with structure
# ----------------------------------------------------------------------

# Scene vocab the mock/real scene classifier can emit; we paint frames so a
# video has a dominant scene, and bias queries toward it.
_SCENES = ["indoor", "outdoor", "nature", "urban", "beach", "sport"]
_QUERY_TEMPLATES = [
    ("a person walking {scene}", True, False),
    ("an exciting highlight in a {scene} scene", False, True),
    ("a close-up of a face {scene}", True, False),
    ("a calm {scene} view", False, False),
    ("people celebrating at a {scene} party", True, True),
]


def _make_video_frames(rng: np.random.Generator, n_frames: int,
                       hw: int, scene_tint: np.ndarray,
                       has_person: bool, lively: bool) -> np.ndarray:
    """Synthesise a small frame stack with controllable content.

    scene_tint biases color (-> scene classifier); has_person paints a
    skin-tone blob (-> face detector heuristic); lively raises color variance
    (-> highlight model).
    """
    frames = np.zeros((n_frames, hw, hw, 3), dtype=np.uint8)
    for f in range(n_frames):
        base = scene_tint + (rng.standard_normal(3) * (40 if lively else 12))
        img = np.clip(np.tile(base, (hw, hw, 1)), 0, 255).astype(np.uint8)
        if lively:
            img += (rng.integers(0, 60, (hw, hw, 3))).astype(np.uint8)
        if has_person:
            # skin-tone blob: r > g > b, moderate saturation (mock face heuristic)
            cy, cx = rng.integers(hw // 4, 3 * hw // 4, 2)
            img[max(0, cy-6):cy+6, max(0, cx-6):cx+6] = [200, 150, 110]
        frames[f] = np.clip(img, 0, 255).astype(np.uint8)
    return frames


def build_synthetic_dataset(n_videos: int = 80,
                            n_queries: int = 160,
                            n_frames: int = 6,
                            hw: int = 64,
                            use_real_models: bool = False,
                            seed: int = 42) -> Dataset:
    """Construct a frame-level synthetic gallery + queries with no external files."""
    rng = np.random.default_rng(seed)
    bundle = build_model_bundle(use_real=use_real_models)

    # Per-video latent attributes.
    scene_idx = rng.integers(0, len(_SCENES), n_videos)
    has_person = rng.random(n_videos) < 0.5
    lively = rng.random(n_videos) < 0.4
    scene_tints = {i: np.array([60 + 30 * i, 80, 120 - 10 * i], np.float32)
                   for i in range(len(_SCENES))}

    video_ids, frames_per_video = [], []
    for v in range(n_videos):
        vid = f"vid_{v:04d}"
        video_ids.append(vid)
        frames_per_video.append(_make_video_frames(
            rng, n_frames, hw, scene_tints[int(scene_idx[v])],
            bool(has_person[v]), bool(lively[v])))

    if False:  # placeholder marker; real models would log progress
        pass
    gallery = extract_gallery_signals(bundle, video_ids, frames_per_video,
                                      verbose=False)

    # Queries: pick a GT video, build a text that matches its attributes so the
    # right experts carry signal, and embed the text with the bundle's encoder.
    gt_idx = rng.integers(0, n_videos, n_queries)
    query_texts, gt_video_ids, priors = [], [], []
    for q in range(n_queries):
        gi = int(gt_idx[q])
        scene = _SCENES[int(scene_idx[gi])]
        tmpl, _wants_p, _wants_h = _QUERY_TEMPLATES[q % len(_QUERY_TEMPLATES)]
        text = tmpl.format(scene=scene)
        query_texts.append(text)
        gt_video_ids.append(video_ids[gi])
        temb = bundle.clip.encode_text([text])[0]
        priors.append(build_query_priors(text, temb))

    return Dataset(gallery=gallery, video_ids=video_ids,
                   query_texts=query_texts, query_priors=priors,
                   gt_video_ids=gt_video_ids, bundle=bundle)


# __APPEND_REAL_LOADER__


# ----------------------------------------------------------------------
#  Real video gallery loader
# ----------------------------------------------------------------------

def _decode_frames(video_path: str, n_frames: int, hw: int) -> np.ndarray:
    """Uniformly sample n_frames RGB frames from a video file (needs OpenCV)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames
    idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (hw, hw))
        frames.append(frame.astype(np.uint8))
    cap.release()
    if not frames:
        frames = [np.zeros((hw, hw, 3), np.uint8)]
    return np.stack(frames, axis=0)


def load_video_dataset(videos_dir: str,
                       queries_csv: str,
                       n_frames: int = 8,
                       hw: int = 224,
                       use_real_models: bool = True,
                       seed: int = 42) -> Dataset:
    """Real gallery: decode frames from videos, run the 5 experts over them.

    Args:
        videos_dir: directory of .mp4 files (or with a manifest.json of
            ``[{"id":..., "path":...}, ...]``). Video ids are filenames' stems
            unless a manifest provides ids.
        queries_csv: CSV with columns ``sentence`` and ``video_id``.
        use_real_models: True -> real backbones (needs weights); falls back to
            mocks on failure.
    """
    bundle = build_model_bundle(use_real=use_real_models)

    vdir = Path(videos_dir)
    manifest = vdir / "manifest.json"
    if manifest.exists():
        import json
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        entries = [{"id": p.stem, "path": str(p)}
                   for p in sorted(vdir.glob("*.mp4"))]

    video_ids, frames_per_video = [], []
    for e in entries:
        video_ids.append(str(e["id"]))
        frames_per_video.append(_decode_frames(str(e["path"]), n_frames, hw))
    gallery = extract_gallery_signals(bundle, video_ids, frames_per_video,
                                      verbose=True)

    query_texts, gt_video_ids = [], []
    with open(queries_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            query_texts.append(row["sentence"])
            gt_video_ids.append(str(row["video_id"]))

    priors = []
    embs = bundle.clip.encode_text(query_texts)
    for text, temb in zip(query_texts, embs):
        priors.append(build_query_priors(text, temb))

    return Dataset(gallery=gallery, video_ids=video_ids,
                   query_texts=query_texts, query_priors=priors,
                   gt_video_ids=gt_video_ids, bundle=bundle)
