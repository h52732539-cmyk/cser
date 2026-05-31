"""Run the 5 frozen expert models over a video gallery -> per-video signals.

This is the bridge between the real models in ``tasks/real_models.py`` (mock
fallback in ``tasks/mock_models.py``) and the CSER value function. For every
gallery video we sample frames once and cache each expert's aggregated output;
at query time the experts re-score cheaply against these cached signals (no
re-running the models per query — exactly the offline-index philosophy of the
repo).

Per-video signals produced (aggregated over the video's sampled frames):

    semantic   : (n_frames, D) CLIP frame embeddings -> stored as mean + protos
    highlight  : scalar max saliency (MomentDETR)        in [0, 1]
    face       : scalar max face confidence (SCRFD)      in [0, 1]
    face_id    : (D,) mean ArcFace embedding over face frames
    scene      : dict{label: fraction} scene distribution (MobileNetV3)

How each optional expert reranks the gallery for a query (see retrieval.py):

    highlight  -> videos with strong highlights get boosted (query-agnostic prior)
    face       -> if the query implies a person, boost videos that contain faces
    face_id    -> rerank by face-embedding similarity to the query's face prior
    scene      -> boost videos whose dominant scene matches the query's scene cue

The ``ModelBundle`` holds the 5 model objects; build it with real or mock
models via :func:`build_model_bundle`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
#  Model bundle (real or mock)
# ----------------------------------------------------------------------

@dataclass
class ModelBundle:
    clip: object          # .encode_frames(list[img]) , .encode_text(list[str])
    highlight: object     # .score(list[img]) -> list[float]
    face_det: object      # .detect(list[img]) -> list[(bool, conf)]
    face_emb: object      # .embed(list[img]) -> list[vec]
    scene: object         # .classify(list[img]) -> list[label]


def build_model_bundle(use_real: bool = False) -> ModelBundle:
    """Construct the 5 experts; fall back to mocks if real weights are missing.

    Mirrors demo/run_benchmark_v2.py::build_models so behaviour is consistent
    with the rest of the repo.
    """
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if use_real:
        try:
            from tasks import real_models
            b = ModelBundle(
                clip=real_models.RealCLIPModel(),
                highlight=real_models.MomentDETRHighlightModel(),
                face_det=real_models.InsightFaceDetector(),
                face_emb=real_models.InsightFaceEmbedder(),
                scene=real_models.MobileNetV3SceneClassifier(),
            )
            print("[cser] using REAL expert models")
            return b
        except Exception as e:                       # missing weights / deps
            print(f"[cser] real-model init failed ({e}); falling back to mocks")

    from tasks import (MockCLIPModel, MockHighlightModel, MockFaceDetector,
                       MockFaceEmbedder, MockSceneClassifier)
    print("[cser] using MOCK expert models")
    return ModelBundle(
        clip=MockCLIPModel(dim=128),
        highlight=MockHighlightModel(),
        face_det=MockFaceDetector(),
        face_emb=MockFaceEmbedder(dim=64),
        scene=MockSceneClassifier(),
    )


# ----------------------------------------------------------------------
#  Per-video expert signals
# ----------------------------------------------------------------------

@dataclass
class VideoExpertSignals:
    """Cached aggregated expert outputs for one gallery video."""
    video_id: str
    clip_mean: np.ndarray                 # (D,) L2-normalised mean frame embedding
    highlight_score: float                # max saliency in [0, 1]
    face_score: float                     # max face confidence in [0, 1]
    face_emb: np.ndarray                  # (Df,) mean ArcFace embedding (zeros if no face)
    scene_dist: Dict[str, float]          # {label: fraction of frames}


@dataclass
class GallerySignals:
    """Expert signals for the whole gallery, in video order."""
    video_ids: List[str]
    signals: List[VideoExpertSignals]
    clip_dim: int
    face_dim: int

    def __post_init__(self):
        self._idx = {v: i for i, v in enumerate(self.video_ids)}

    @property
    def size(self) -> int:
        return len(self.video_ids)

    # vectorised accessors (length N) ---------------------------------
    def clip_matrix(self) -> np.ndarray:
        return np.stack([s.clip_mean for s in self.signals], axis=0)

    def highlight_vector(self) -> np.ndarray:
        return np.array([s.highlight_score for s in self.signals], dtype=np.float32)

    def face_vector(self) -> np.ndarray:
        return np.array([s.face_score for s in self.signals], dtype=np.float32)

    def face_emb_matrix(self) -> np.ndarray:
        return np.stack([s.face_emb for s in self.signals], axis=0)

    def scene_label_of(self, i: int) -> str:
        d = self.signals[i].scene_dist
        return max(d, key=d.get) if d else "other"

    def subset(self, keep_ids: Sequence[str]) -> "GallerySignals":
        """Return a new GallerySignals containing only ``keep_ids`` (in order)."""
        idx = [self._idx[v] for v in keep_ids if v in self._idx]
        return GallerySignals(
            video_ids=[self.video_ids[i] for i in idx],
            signals=[self.signals[i] for i in idx],
            clip_dim=self.clip_dim, face_dim=self.face_dim)


# __APPEND_EXTRACTOR__


# ----------------------------------------------------------------------
#  Extraction: run the 5 models over each video's frames once
# ----------------------------------------------------------------------

def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def extract_gallery_signals(bundle: ModelBundle,
                            video_ids: Sequence[str],
                            frames_per_video: Sequence[np.ndarray],
                            verbose: bool = True) -> GallerySignals:
    """Run all 5 experts over each video's frames -> cached per-video signals.

    Args:
        bundle: the 5 expert models.
        video_ids: gallery video ids.
        frames_per_video: for each video, an array (n_frames, H, W, 3) uint8.
            (Synthetic galleries can pass small random frames; real galleries
            pass decoded keyframes.)
    """
    signals: List[VideoExpertSignals] = []
    clip_dim = getattr(bundle.clip, "dim", 512)
    face_dim = getattr(bundle.face_emb, "dim", 512)

    for vi, (vid, frames) in enumerate(zip(video_ids, frames_per_video)):
        imgs = list(frames)
        # --- semantic (CLIP) ---
        embs = bundle.clip.encode_frames(imgs)
        emb_mat = np.stack(embs, axis=0) if embs else np.zeros((1, clip_dim), np.float32)
        clip_mean = _l2(emb_mat.mean(axis=0))

        # --- highlight (MomentDETR) ---
        hl = bundle.highlight.score(imgs)
        highlight_score = float(max(hl)) if hl else 0.0

        # --- face detect (SCRFD) ---
        dets = bundle.face_det.detect(imgs)
        face_conf = [c for (has, c) in dets if has]
        face_score = float(max(face_conf)) if face_conf else 0.0

        # --- face embed (ArcFace) over face-bearing frames ---
        face_frames = [imgs[k] for k, (has, _) in enumerate(dets) if has]
        if face_frames:
            fembs = bundle.face_emb.embed(face_frames)
            face_emb = _l2(np.stack(fembs, axis=0).mean(axis=0))
        else:
            face_emb = np.zeros(face_dim, dtype=np.float32)

        # --- scene (MobileNetV3) ---
        labels = bundle.scene.classify(imgs)
        scene_dist: Dict[str, float] = {}
        if labels:
            for lb in labels:
                scene_dist[lb] = scene_dist.get(lb, 0.0) + 1.0 / len(labels)

        signals.append(VideoExpertSignals(
            video_id=vid, clip_mean=clip_mean.astype(np.float32),
            highlight_score=highlight_score, face_score=face_score,
            face_emb=face_emb.astype(np.float32), scene_dist=scene_dist,
        ))
        if verbose and (vi + 1) % 100 == 0:
            print(f"  [expert-extract {vi+1}/{len(video_ids)}]")

    return GallerySignals(video_ids=list(video_ids), signals=signals,
                          clip_dim=clip_dim, face_dim=face_dim)


# ----------------------------------------------------------------------
#  Query-side expert priors (cheap, per-query)
# ----------------------------------------------------------------------

@dataclass
class QueryExpertPriors:
    """Per-query signals the optional experts use to rerank the gallery.

    Derived from the query text + the query's CLIP embedding. These let each
    optional expert produce a *query-conditioned* per-video score, which is what
    makes its marginal value depend on the query (and on the other experts).
    """
    text_emb: np.ndarray                  # (D,) CLIP text embedding
    wants_person: bool                    # query mentions a person/face
    wants_highlight: bool                 # query mentions an action/highlight
    scene_cue: Optional[str]              # query's scene label, if any
    face_emb: Optional[np.ndarray] = None # face-id prior, if a reference is given


_PERSON_WORDS = ("person", "man", "woman", "people", "face", "someone",
                 "child", "boy", "girl", "他", "她", "人", "脸", "面孔")
_HIGHLIGHT_WORDS = ("jump", "dance", "score", "goal", "trick", "highlight",
                    "action", "running", "fast", "celebrate", "精彩", "高光")
_SCENE_WORDS = {
    "indoor": ("indoor", "room", "inside", "室内", "屋里"),
    "outdoor": ("outdoor", "outside", "户外"),
    "nature": ("nature", "forest", "mountain", "natural", "自然", "山"),
    "urban": ("city", "urban", "street", "downtown", "城市", "街"),
    "beach": ("beach", "coast", "sea", "海", "沙滩"),
    "sport": ("sport", "game", "match", "运动", "比赛"),
    "kitchen": ("kitchen", "cook", "厨房"),
    "office": ("office", "work", "办公"),
    "party": ("party", "birthday", "聚会", "派对"),
    "street": ("street", "road", "街道", "马路"),
    "vehicle": ("car", "drive", "vehicle", "车", "驾驶"),
}


def build_query_priors(query_text: str, text_emb: np.ndarray,
                       face_emb: Optional[np.ndarray] = None) -> QueryExpertPriors:
    q = query_text.lower()
    scene_cue = None
    for label, words in _SCENE_WORDS.items():
        if any(w in q for w in words):
            scene_cue = label
            break
    return QueryExpertPriors(
        text_emb=_l2(np.asarray(text_emb, np.float32)),
        wants_person=any(w in q for w in _PERSON_WORDS),
        wants_highlight=any(w in q for w in _HIGHLIGHT_WORDS),
        scene_cue=scene_cue,
        face_emb=face_emb,
    )
