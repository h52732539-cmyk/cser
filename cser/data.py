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
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .expert_features import (ModelBundle, build_model_bundle,
                              extract_gallery_signals, GallerySignals,
                              build_query_priors, QueryExpertPriors,
                              VideoExpertSignals)


@dataclass
class Dataset:
    gallery: GallerySignals               # cached per-video expert signals
    video_ids: List[str]
    query_texts: List[str]
    query_priors: List[QueryExpertPriors]
    gt_video_ids: List[str]
    bundle: ModelBundle
    n_videos_total: Optional[int] = None
    failed_video_ids: List[str] = None
    cache_manifest: Optional[Dict] = None

    def __post_init__(self):
        if self.n_videos_total is None:
            self.n_videos_total = self.gallery.size
        if self.failed_video_ids is None:
            self.failed_video_ids = []

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
                   gt_video_ids=gt_video_ids, bundle=bundle,
                   n_videos_total=n_videos, failed_video_ids=[])


# __APPEND_REAL_LOADER__


# ----------------------------------------------------------------------
#  Real video gallery loader
# ----------------------------------------------------------------------

def _decode_frames(video_path: str, n_frames: int, hw: int) -> np.ndarray:
    """Uniformly sample n_frames RGB frames from a video file (needs OpenCV)."""
    import cv2
    if not Path(video_path).exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open video: {video_path}")
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
        raise RuntimeError(f"could not decode frames: {video_path}")
    return np.stack(frames, axis=0)


def _cache_paths(cache_dir: Optional[str]) -> Tuple[Optional[Path], Optional[Path]]:
    if cache_dir is None:
        return None, None
    p = Path(cache_dir)
    return p / "gallery_cache.npz", p / "manifest.json"


def _bundle_manifest(bundle: ModelBundle, use_real_models: bool) -> Dict:
    parts = {
        "clip": bundle.clip,
        "highlight": bundle.highlight,
        "face_det": bundle.face_det,
        "face_emb": bundle.face_emb,
        "scene": bundle.scene,
    }
    ckpts = {}
    for name, obj in parts.items():
        for attr in ("checkpoint_path", "ckpt", "ckpt_path", "model_dir", "root"):
            if hasattr(obj, attr):
                ckpts[name] = str(getattr(obj, attr))
                break
        else:
            ckpts[name] = None
    return {
        "expert_class_names": {
            k: (type(v).__name__ if v is not None else None)
            for k, v in parts.items()
        },
        "checkpoint_paths": ckpts,
        "mock_fallback_occurred": not use_real_models,
    }


def _write_gallery_cache(cache_dir: str,
                         gallery: GallerySignals,
                         n_videos_total: int,
                         failed_video_ids: Sequence[str],
                         bundle: ModelBundle,
                         use_real_models: bool,
                         complete: bool = True) -> Dict:
    cache_path, manifest_path = _cache_paths(cache_dir)
    assert cache_path is not None and manifest_path is not None
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    clip = gallery.clip_matrix().astype(np.float32)
    highlight = gallery.highlight_vector().astype(np.float32)
    face = gallery.face_vector().astype(np.float32)
    face_emb = gallery.face_emb_matrix().astype(np.float32)
    scene_dist_json = np.array([
        json.dumps(s.scene_dist, sort_keys=True) for s in gallery.signals
    ])
    cache_tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with cache_tmp.open("wb") as f:
        np.savez_compressed(
            f,
            video_ids=np.array(gallery.video_ids),
            clip_mean=clip,
            highlight=highlight,
            face=face,
            face_emb=face_emb,
            scene_dist_json=scene_dist_json,
            clip_dim=np.array([gallery.clip_dim], dtype=np.int32),
            face_dim=np.array([gallery.face_dim], dtype=np.int32),
        )
    cache_tmp.replace(cache_path)

    manifest = {
        "kind": "cser_gallery_expert_cache",
        "cache_npz": str(cache_path),
        "complete": bool(complete),
        "n_videos_total": int(n_videos_total),
        "n_videos_loaded": int(gallery.size),
        "failed_video_ids": list(failed_video_ids),
        "feature_shapes": {
            "clip_mean": list(clip.shape),
            "highlight": list(highlight.shape),
            "face": list(face.shape),
            "face_emb": list(face_emb.shape),
        },
        "feature_dtype": "float32",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest.update(_bundle_manifest(bundle, use_real_models))
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest


def _read_gallery_cache(cache_dir: str) -> Tuple[GallerySignals, Dict]:
    cache_path, manifest_path = _cache_paths(cache_dir)
    assert cache_path is not None and manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    d = np.load(cache_path)
    video_ids = [str(v) for v in d["video_ids"].tolist()]
    scene_json = [str(x) for x in d["scene_dist_json"].tolist()]
    clip = d["clip_mean"].astype(np.float32)
    highlight = d["highlight"].astype(np.float32)
    face = d["face"].astype(np.float32)
    face_emb = d["face_emb"].astype(np.float32)
    signals = []
    for i, vid in enumerate(video_ids):
        signals.append(VideoExpertSignals(
            video_id=vid,
            clip_mean=clip[i],
            highlight_score=float(highlight[i]),
            face_score=float(face[i]),
            face_emb=face_emb[i],
            scene_dist=json.loads(scene_json[i]),
        ))
    gallery = GallerySignals(
        video_ids=video_ids,
        signals=signals,
        clip_dim=int(d["clip_dim"][0]),
        face_dim=int(d["face_dim"][0]),
    )
    return gallery, manifest


def load_video_dataset(videos_dir: str,
                       queries_csv: str,
                       n_frames: int = 8,
                       hw: int = 224,
                       use_real_models: bool = True,
                       cache_dir: Optional[str] = None,
                       seed: int = 42) -> Dataset:
    """Real gallery: decode frames from videos, run the 5 experts over them.

    Args:
        videos_dir: directory of .mp4 files (or with a manifest.json of
            ``[{"id":..., "path":...}, ...]``). Video ids are filenames' stems
            unless a manifest provides ids.
        queries_csv: CSV with columns ``sentence`` and ``video_id``.
        use_real_models: True -> real backbones (needs weights); raises on
            initialization failure so diagnostic runs cannot silently use mocks.
    """
    vdir = Path(videos_dir)
    manifest = vdir / "manifest.json"
    if manifest.exists():
        import json
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        entries = [{"id": p.stem, "path": str(p)}
                   for p in sorted(vdir.glob("*.mp4"))]

    cache_path, manifest_path = _cache_paths(cache_dir)
    cache_manifest = None
    gallery = None
    failed_ids = set()
    entry_ids = [str(e["id"]) for e in entries]
    entry_id_set = set(entry_ids)
    if (cache_path is not None and manifest_path is not None
            and cache_path.exists() and manifest_path.exists()):
        gallery, cache_manifest = _read_gallery_cache(cache_dir)
        unknown_ids = set(gallery.video_ids) - entry_id_set
        if unknown_ids:
            sample = sorted(unknown_ids)[:5]
            raise RuntimeError(
                f"gallery cache contains video ids absent from the current "
                f"manifest: {sample}"
            )
        failed_ids.update(
            str(v) for v in cache_manifest.get("failed_video_ids", [])
        )
        inferred_complete = (
            gallery.size + len(failed_ids) >= len(entries)
        )
        cache_complete = bool(
            cache_manifest.get("complete", inferred_complete)
        )
        state = "complete" if cache_complete else "partial"
        print(f"[cser] loaded {state} gallery cache {cache_path} "
              f"({gallery.size}/{len(entries)} videos)")
    else:
        cache_complete = False

    bundle = build_model_bundle(
        use_real=use_real_models,
        text_only=cache_complete,
    )

    if not cache_complete:
        cached_ids = set(gallery.video_ids) if gallery is not None else set()
        pending = [e for e in entries if str(e["id"]) not in cached_ids]
        checkpoint_interval = max(
            1, int(os.environ.get("CSER_CACHE_CHECKPOINT_INTERVAL", "100"))
        )
        print(f"[cser] extracting {len(pending)} uncached videos "
              f"(checkpoint interval={checkpoint_interval})")

        for start in range(0, len(pending), checkpoint_interval):
            chunk = pending[start:start + checkpoint_interval]
            chunk_ids, frames_per_video = [], []
            for e in chunk:
                vid = str(e["id"])
                try:
                    frames = _decode_frames(str(e["path"]), n_frames, hw)
                except Exception as exc:
                    failed_ids.add(vid)
                    print(f"[cser] failed video {vid}: {exc}")
                    continue
                failed_ids.discard(vid)
                chunk_ids.append(vid)
                frames_per_video.append(frames)

            if chunk_ids:
                chunk_gallery = extract_gallery_signals(
                    bundle, chunk_ids, frames_per_video, verbose=False
                )
                if gallery is None:
                    gallery = chunk_gallery
                else:
                    gallery = GallerySignals(
                        video_ids=gallery.video_ids + chunk_gallery.video_ids,
                        signals=gallery.signals + chunk_gallery.signals,
                        clip_dim=gallery.clip_dim,
                        face_dim=gallery.face_dim,
                    )

            failed_video_ids = [
                vid for vid in entry_ids if vid in failed_ids
            ]
            if cache_dir is not None and gallery is not None:
                cache_manifest = _write_gallery_cache(
                    cache_dir, gallery, len(entries), failed_video_ids,
                    bundle, use_real_models, complete=False)
            loaded = gallery.size if gallery is not None else 0
            print(f"[cser] expert cache checkpoint "
                  f"loaded={loaded}/{len(entries)} "
                  f"failed={len(failed_video_ids)}")

        if gallery is None:
            raise RuntimeError("no videos could be decoded and cached")
        failed_video_ids = [vid for vid in entry_ids if vid in failed_ids]
        if cache_dir is not None:
            cache_manifest = _write_gallery_cache(
                cache_dir, gallery, len(entries), failed_video_ids,
                bundle, use_real_models, complete=True)
    else:
        failed_video_ids = [vid for vid in entry_ids if vid in failed_ids]

    video_ids = list(gallery.video_ids)
    print(f"N videos failed: {len(failed_video_ids)}; "
          f"N videos loaded: {gallery.size}; N videos total: {len(entries)}")

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
                   gt_video_ids=gt_video_ids, bundle=bundle,
                   n_videos_total=int((cache_manifest or {}).get(
                       "n_videos_total", len(entries))),
                   failed_video_ids=failed_video_ids,
                   cache_manifest=cache_manifest)
