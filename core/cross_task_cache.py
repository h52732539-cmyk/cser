"""Cross-task embedding cache.

Every frame that enters the system receives a `FrameIdentity`. Any
upstream model (Huawei CLIP, MomentDETR CLIP, MobileNetV3, InsightFace)
can ask the cache for its prior output on a perceptually equivalent
frame and, on a hit, skip the call entirely.

Key invariants:

  * The cache never alters the model call itself — when we call the
    model, we call it exactly like before.
  * A cache hit is always a *replay*, never a new computation. So for
    a properly-keyed cache level (byte_hash), outputs are bit-exact.
  * Perceptual hits are gated by an explicit `mode` parameter, so the
    caller (task adapter) chooses the accuracy-vs-hit-rate trade-off.

Usage:
    cache = CrossTaskCache()
    emb = cache.get_or_compute(
        fid, model_id="huawei_clip",
        compute_fn=lambda: huawei_clip.encode_image([frame])[0],
        mode="byte",             # or "phash", "embedding"
    )
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .frame_identity import FrameIdentity, hamming, embedding_cosine


# ----------------------------------------------------------------------
#  Per-model sub-cache
# ----------------------------------------------------------------------

class _ModelCache:
    """Byte-exact LRU with an optional pHash secondary index."""

    def __init__(self, max_size: int = 4096) -> None:
        self._byte: "OrderedDict[bytes, Any]" = OrderedDict()
        self._phash_idx: Dict[int, bytes] = {}
        self.max_size = max_size
        self.hits_byte = 0
        self.hits_phash = 0
        self.hits_emb = 0
        self.misses = 0

    def _evict_if_needed(self) -> None:
        while len(self._byte) > self.max_size:
            k, _ = self._byte.popitem(last=False)
            # remove reverse index entries pointing to this key
            to_drop = [p for p, kk in self._phash_idx.items() if kk == k]
            for p in to_drop:
                self._phash_idx.pop(p, None)

    def get(self,
            fid: FrameIdentity,
            mode: str = "byte",
            bits_tol: int = 5,
            emb_cos_tol: float = 0.97,
            emb_lookup_cb: Optional[Callable[[bytes], Optional[np.ndarray]]] = None,
            ) -> Optional[Any]:
        """Retrieve a cached value under the specified matching mode."""
        if fid.byte_key in self._byte:
            self._byte.move_to_end(fid.byte_key)
            self.hits_byte += 1
            return self._byte[fid.byte_key]

        if mode in ("phash", "embedding"):
            # search pHash index for nearest neighbour within bits_tol
            best_key: Optional[bytes] = None
            best_dist = bits_tol + 1
            for p, k in self._phash_idx.items():
                d = hamming(p, fid.phash64)
                if d < best_dist:
                    best_dist = d
                    best_key = k
                    if d == 0:
                        break
            if best_key is not None and best_dist <= bits_tol:
                self._byte.move_to_end(best_key)
                self.hits_phash += 1
                return self._byte[best_key]

        # embedding-level matching is an escape hatch for the retrieval
        # pipeline — callers must pre-populate fid.embedding.
        if mode == "embedding" and fid.embedding is not None and emb_lookup_cb is not None:
            for bk in reversed(self._byte.keys()):
                other = emb_lookup_cb(bk)
                if other is None:
                    continue
                if embedding_cosine(fid.embedding, other) >= emb_cos_tol:
                    self._byte.move_to_end(bk)
                    self.hits_emb += 1
                    return self._byte[bk]

        self.misses += 1
        return None

    def put(self, fid: FrameIdentity, value: Any) -> None:
        if fid.byte_key in self._byte:
            self._byte.move_to_end(fid.byte_key)
            self._byte[fid.byte_key] = value
            return
        self._byte[fid.byte_key] = value
        self._phash_idx[fid.phash64] = fid.byte_key
        self._evict_if_needed()

    def clear(self) -> None:
        self._byte.clear()
        self._phash_idx.clear()
        self.hits_byte = self.hits_phash = self.hits_emb = self.misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "hits_byte": self.hits_byte,
            "hits_phash": self.hits_phash,
            "hits_emb": self.hits_emb,
            "misses": self.misses,
            "size": len(self._byte),
        }


# ----------------------------------------------------------------------
#  Public CrossTaskCache
# ----------------------------------------------------------------------

@dataclass
class CrossTaskCache:
    """Cache keyed by (model_id, frame_identity)."""

    max_size_per_model: int = 4096
    _pools: Dict[str, _ModelCache] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def _pool(self, model_id: str) -> _ModelCache:
        p = self._pools.get(model_id)
        if p is None:
            p = _ModelCache(max_size=self.max_size_per_model)
            self._pools[model_id] = p
        return p

    def get(self,
            fid: FrameIdentity,
            model_id: str,
            mode: str = "byte",
            bits_tol: int = 5,
            emb_cos_tol: float = 0.97) -> Optional[Any]:
        return self._pool(model_id).get(
            fid, mode=mode, bits_tol=bits_tol, emb_cos_tol=emb_cos_tol,
        )

    def put(self, fid: FrameIdentity, model_id: str, value: Any) -> None:
        self._pool(model_id).put(fid, value)

    def get_or_compute(self,
                       fid: FrameIdentity,
                       model_id: str,
                       compute_fn: Callable[[], Any],
                       mode: str = "byte",
                       bits_tol: int = 5,
                       emb_cos_tol: float = 0.97) -> Tuple[Any, bool]:
        """Returns (value, was_cached)."""
        hit = self.get(fid, model_id=model_id, mode=mode,
                       bits_tol=bits_tol, emb_cos_tol=emb_cos_tol)
        if hit is not None:
            return hit, True
        val = compute_fn()
        self.put(fid, model_id, val)
        return val, False

    def clear(self) -> None:
        for p in self._pools.values():
            p.clear()

    def stats(self) -> Dict[str, Dict[str, int]]:
        return {mid: p.stats for mid, p in self._pools.items()}

    def total_savings(self) -> Dict[str, int]:
        hits = sum(p.hits_byte + p.hits_phash + p.hits_emb
                   for p in self._pools.values())
        miss = sum(p.misses for p in self._pools.values())
        total = hits + miss
        return {
            "hits_total": hits,
            "misses": miss,
            "hit_rate_pct": int(100 * hits / max(total, 1)),
            "calls_saved": hits,
        }
