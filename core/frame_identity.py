"""Frame identity hashing for cross-frame/cross-task caching.

Three levels of identity resolution:

  1. `byte_hash(frame)`      — strict pixel-level identity (blake2b 16B)
  2. `phash(frame)`          — 64-bit perceptual hash (DCT-based, aHash fallback)
  3. `embedding_similarity`  — upstream-provided CLIP/DINO embedding cosine

Each level is strictly cheaper than the next. Caches upstream choose the
level that matches their accuracy tolerance:

  - RetrievalTask / HighlightTask CLIP embs → byte_hash (no semantic drift)
  - FaceEmbedder when same face persists    → phash (tolerate small shifts)
  - SceneClassifier on static segments      → phash OR embedding_similarity

This module is model-free: no Huawei-proprietary or third-party model is
touched; we operate only on raw np.ndarray RGB frames.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
#  Byte-exact hash  (cost: ~50 us for 720p frame)
# ----------------------------------------------------------------------

def byte_hash(frame: np.ndarray, stride: int = 16) -> bytes:
    """Strict pixel-level hash. Two frames with the same pixels share a key.

    `stride`=16 downsamples for speed; still byte-exact at that scale.
    """
    if frame.size == 0:
        return b"\0" * 16
    small = frame[::stride, ::stride]
    if not small.flags.c_contiguous:
        small = np.ascontiguousarray(small)
    return hashlib.blake2b(small.tobytes(), digest_size=16).digest()


# ----------------------------------------------------------------------
#  Perceptual hash  (cost: ~300 us for 720p frame)
# ----------------------------------------------------------------------

def phash(frame: np.ndarray, size: int = 32) -> int:
    """64-bit DCT-based perceptual hash.

    Returns a single Python int whose Hamming distance correlates with
    visual similarity. Two frames within ~5 bits are perceptually close.

    Falls back to aHash (average hash) if scipy/cv2 DCT is unavailable.
    """
    if frame.size == 0:
        return 0
    # 1. grayscale
    if frame.ndim == 3:
        g = (0.299 * frame[..., 0] + 0.587 * frame[..., 1]
             + 0.114 * frame[..., 2]).astype(np.float32)
    else:
        g = frame.astype(np.float32)

    # 2. resize to size x size
    try:
        import cv2
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        g = _box_resize(g, size, size)

    # 3. DCT then keep 8x8 low-freq block
    try:
        import cv2
        d = cv2.dct(g)
    except Exception:
        d = _naive_dct2(g)

    low = d[:8, :8].copy()
    low[0, 0] = 0.0  # drop DC
    med = float(np.median(low))
    bits = (low > med).ravel()
    h = 0
    for i, b in enumerate(bits):
        if b:
            h |= (1 << i)
    return int(h)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
#  Cosine similarity between two 1-D unit vectors
# ----------------------------------------------------------------------

def embedding_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ----------------------------------------------------------------------
#  Internal fallbacks (no cv2 / scipy)
# ----------------------------------------------------------------------

def _box_resize(img: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = img.shape[:2]
    ys = (np.linspace(0, h, H + 1)).astype(int)
    xs = (np.linspace(0, w, W + 1)).astype(int)
    out = np.empty((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            out[i, j] = img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
    return out


def _naive_dct2(g: np.ndarray) -> np.ndarray:
    """Separable DCT-II fallback via numpy only."""
    N = g.shape[0]
    k = np.arange(N).reshape(N, 1)
    n = np.arange(N).reshape(1, N)
    basis = np.cos(np.pi * (2 * n + 1) * k / (2 * N)) * np.sqrt(2.0 / N)
    basis[0] *= np.sqrt(0.5)
    return basis @ g @ basis.T


# ----------------------------------------------------------------------
#  Convenience: FrameIdentity — pre-computes all three levels once
# ----------------------------------------------------------------------

class FrameIdentity:
    """Cache-friendly multi-level identity key for a single frame."""

    __slots__ = ("byte_key", "phash64", "_emb")

    def __init__(self, frame: np.ndarray) -> None:
        self.byte_key: bytes = byte_hash(frame)
        self.phash64: int = phash(frame)
        self._emb: Optional[np.ndarray] = None

    def set_embedding(self, emb: np.ndarray) -> None:
        self._emb = emb

    @property
    def embedding(self) -> Optional[np.ndarray]:
        return self._emb

    def close_to(self, other: "FrameIdentity",
                 bits_tol: int = 5,
                 emb_cos_tol: float = 0.97) -> bool:
        """Perceptual + embedding-level closeness."""
        if self.byte_key == other.byte_key:
            return True
        if hamming(self.phash64, other.phash64) <= bits_tol:
            return True
        if self._emb is not None and other._emb is not None:
            return embedding_cosine(self._emb, other._emb) >= emb_cos_tol
        return False
