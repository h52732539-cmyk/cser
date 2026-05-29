"""Deterministic mock models for the benchmark.

Each mock model produces reproducible outputs as a function of the input
frame's pixel content. This makes the benchmark completely self-contained:
no heavy weights to download, yet the accuracy comparison across strategies
remains meaningful because the same frame always yields the same output.
"""
from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np


def _deterministic_embedding(
    frame: np.ndarray, dim: int = 128, seed: int = 0
) -> np.ndarray:
    """Hash-based deterministic embedding of a frame.

    Uses a small SHA256 of a downsampled pixel grid -> numpy RNG -> vector.
    Same input frame -> same output vector.
    """
    small = frame[::16, ::16, :]  # aggressive downsample
    h = hashlib.sha256(small.tobytes()).digest()
    s = int.from_bytes(h[:8], "big") ^ seed
    rng = np.random.default_rng(s)
    # Blend some perceptual signal in so that "similar" frames produce
    # "similar" embeddings.
    mean_rgb = small.reshape(-1, 3).mean(axis=0) / 255.0
    v = rng.standard_normal(dim).astype(np.float32)
    v[:3] += (mean_rgb - 0.5) * 4.0
    v /= np.linalg.norm(v) + 1e-8
    return v


# ----------------------------------------------------------------------
#  Mock CLIP
# ----------------------------------------------------------------------

class MockCLIPModel:
    """Produces a deterministic 128-D embedding per frame."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode_frames(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [_deterministic_embedding(img, self.dim) for img in images]

    def encode_text(self, texts: List[str]) -> List[np.ndarray]:
        out = []
        for s in texts:
            h = hashlib.sha256(s.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            out.append(v)
        return out


def make_query_embeddings(n: int = 3, dim: int = 128) -> List[np.ndarray]:
    clip = MockCLIPModel(dim)
    queries = [f"query_{i}" for i in range(n)]
    return clip.encode_text(queries)


# ----------------------------------------------------------------------
#  Mock highlight detector
# ----------------------------------------------------------------------

class MockHighlightModel:
    """Score in [0,1], derived from the frame's color variance.

    High variance (cluttered / colorful) frames score higher -> proxy for
    'interesting' moments.
    """

    def score(self, images: List[np.ndarray]) -> List[float]:
        scores: List[float] = []
        for img in images:
            sub = img[::8, ::8, :].astype(np.float32) / 255.0
            std = float(sub.std())
            s = max(0.0, min(1.0, (std - 0.10) / 0.30))
            scores.append(s)
        return scores


# ----------------------------------------------------------------------
#  Mock face detector / embedder
# ----------------------------------------------------------------------

class MockFaceDetector:
    """Returns (has_face: bool, confidence: float) per frame.

    Heuristic: frames whose hue mean falls in a 'skin tone' band and that
    have moderate saturation are marked as containing a face.
    """

    def detect(self, images: List[np.ndarray]) -> List[Tuple[bool, float]]:
        out: List[Tuple[bool, float]] = []
        for img in images:
            small = img[::8, ::8, :].astype(np.float32) / 255.0
            r = small[..., 0].mean()
            g = small[..., 1].mean()
            b = small[..., 2].mean()
            mx = max(r, g, b)
            mn = min(r, g, b)
            sat = (mx - mn) / (mx + 1e-6)
            skin_like = (r > g > b) and (0.15 < sat < 0.55) and (0.3 < r < 0.8)
            conf = 0.80 + 0.15 * min(1.0, sat) if skin_like else 0.10
            out.append((skin_like, conf))
        return out


class MockFaceEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [_deterministic_embedding(img, self.dim, seed=777)
                for img in images]


# ----------------------------------------------------------------------
#  Mock scene classifier
# ----------------------------------------------------------------------

class MockSceneClassifier:
    """Outputs a scene label from a small fixed vocabulary."""

    VOCAB = ["indoor", "outdoor", "nature", "urban", "sport", "party"]

    def classify(self, images: List[np.ndarray]) -> List[str]:
        labels: List[str] = []
        for img in images:
            h = hashlib.md5(img[::32, ::32, :].tobytes()).digest()
            idx = h[0] % len(self.VOCAB)
            labels.append(self.VOCAB[idx])
        return labels
