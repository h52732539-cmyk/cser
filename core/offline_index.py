"""Offline video retrieval index.

The core efficiency lever under the black-box-model constraint:

  * At **indexing time** (background / charging), call the frozen
    HuaweiCLIP.encode_image ONCE per keyframe, build multi-K prototypes,
    and persist to disk.
  * At **query time**, call HuaweiCLIP.encode_text ONCE and run NNN +
    column-softmax + QAMP scoring on the cached prototypes. No image
    tower is re-invoked.

Phase-3 additions:
  * `VideoIndexEntry.metadata` — typed VideoMetadata (time/GPS/motion...)
  * `OfflineIndex.search_with_meta` — filter by MetaFilter → rerank by
    semantic → optional soft-score fusion.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adaptive_sampler import _BaseSampler, UniformSampler
from .metadata import VideoMetadata, fill_derived_fields
from .query_parser import QueryIntent
from .meta_filter import MetaFilter, fuse_scores


# ----------------------------------------------------------------------
#  Multi-K prototype construction  (CNPR-compatible)
# ----------------------------------------------------------------------

def build_protos(frame_embs: np.ndarray, K: int) -> np.ndarray:
    """Divide frames into K equal segments and average each. Returns (K, D)."""
    n = frame_embs.shape[0]
    if n == 0:
        return np.zeros((K, frame_embs.shape[1] if frame_embs.ndim > 1 else 1),
                         dtype=np.float32)
    if n <= K:
        reps = np.repeat(frame_embs, int(np.ceil(K / n)), axis=0)[:K]
        return reps.astype(np.float32)
    edges = np.linspace(0, n, K + 1).astype(int)
    out = np.zeros((K, frame_embs.shape[1]), dtype=np.float32)
    for i in range(K):
        s, e = edges[i], edges[i + 1]
        if e > s:
            out[i] = frame_embs[s:e].mean(axis=0)
    out /= (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9)
    return out


# ----------------------------------------------------------------------
#  Per-video index entry
# ----------------------------------------------------------------------

@dataclass
class VideoIndexEntry:
    video_id: str
    video_path: str
    duration: float
    key_ts: List[float] = field(default_factory=list)
    frame_embs: Optional[np.ndarray] = None     # (K_frames, D)
    protos: Dict[int, np.ndarray] = field(default_factory=dict)   # K -> (K, D)
    face_timeline: Optional[np.ndarray] = None  # bool at 1 Hz
    scene_timeline: Optional[List[str]] = None
    metadata: Optional[VideoMetadata] = None    # Phase-3: typed meta
    meta: Dict = field(default_factory=dict)    # free-form; legacy

    def payload_bytes(self) -> int:
        n = 0
        if self.frame_embs is not None:
            n += self.frame_embs.nbytes
        for v in self.protos.values():
            n += v.nbytes
        return n


# ----------------------------------------------------------------------
#  Index builder
# ----------------------------------------------------------------------

class OfflineIndexBuilder:
    """Build a persistent index over a gallery of videos."""

    def __init__(self,
                 image_encoder: Callable[[List[np.ndarray]], np.ndarray],
                 face_detector: Optional[Callable[[List[np.ndarray]], List[Tuple[bool, float]]]] = None,
                 scene_classifier: Optional[Callable[[List[np.ndarray]], List[str]]] = None,
                 sampler: Optional[_BaseSampler] = None,
                 k_values: Sequence[int] = (2, 4, 6),
                 max_keyframes: int = 24) -> None:
        self.image_encoder = image_encoder
        self.face_detector = face_detector
        self.scene_classifier = scene_classifier
        self.sampler = sampler or UniformSampler(fps=1.0, max_samples=max_keyframes)
        self.k_values = tuple(k_values)
        self.max_keyframes = max_keyframes

    # ------------------------------------------------------------------

    def build_one(self, video_path: str, video_id: Optional[str] = None,
                  duration: Optional[float] = None) -> VideoIndexEntry:
        import cv2
        if duration is None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            duration = n / fps if fps > 0 else 0.0

        vid = video_id or Path(video_path).stem
        ts_tags = self.sampler.sample(video_path, duration)
        timestamps = [t for t, _ in ts_tags][:self.max_keyframes]
        frames = self._decode(video_path, timestamps)
        if not frames:
            return VideoIndexEntry(
                video_id=vid, video_path=str(video_path), duration=duration,
            )

        embs = np.asarray(self.image_encoder(frames), dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        # normalize for cosine math downstream
        embs /= (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-9)

        protos = {K: build_protos(embs, K) for K in self.k_values}

        face_tl = None
        if self.face_detector is not None:
            dets = self.face_detector(frames)
            face_tl = np.array([bool(p) for p, _ in dets], dtype=bool)
        scene_tl = None
        if self.scene_classifier is not None:
            scene_tl = list(self.scene_classifier(frames))

        return VideoIndexEntry(
            video_id=vid, video_path=str(video_path), duration=duration,
            key_ts=list(map(float, timestamps)),
            frame_embs=embs, protos=protos,
            face_timeline=face_tl, scene_timeline=scene_tl,
        )

    def build_gallery(self,
                       videos: List[Dict],
                       save_path: Optional[str] = None,
                       progress: bool = True) -> "OfflineIndex":
        entries: List[VideoIndexEntry] = []
        for i, v in enumerate(videos):
            if progress:
                print(f"[index] {i + 1}/{len(videos)}  {v.get('id')}")
            try:
                entry = self.build_one(
                    video_path=v["path"],
                    video_id=v.get("id"),
                    duration=v.get("duration"),
                )
                entries.append(entry)
            except Exception as e:
                print(f"  [warn] failed on {v.get('id')}: {e}")
        idx = OfflineIndex(entries=entries)
        if save_path is not None:
            idx.save(save_path)
        return idx

    # ------------------------------------------------------------------

    @staticmethod
    def _decode(video_path: str, timestamps: List[float]) -> List[np.ndarray]:
        try:
            import cv2
        except Exception:
            return []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        out = []
        try:
            for t in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return out


# ----------------------------------------------------------------------
#  Persistent index + retriever
# ----------------------------------------------------------------------

class OfflineIndex:
    """In-memory index + simple persistence + query-time retrieval."""

    def __init__(self, entries: List[VideoIndexEntry]) -> None:
        self.entries = entries
        self._build_flat_view()

    # ---- persistence -------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.entries, f, protocol=4)
        # optional companion meta for human inspection
        meta = [{
            "video_id": e.video_id,
            "duration": e.duration,
            "n_key": len(e.key_ts),
            "bytes": e.payload_bytes(),
        } for e in self.entries]
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OfflineIndex":
        with open(path, "rb") as f:
            entries = pickle.load(f)
        return cls(entries=entries)

    # ---- retrieval ---------------------------------------------------

    def _build_flat_view(self) -> None:
        self._flat_protos: Dict[int, np.ndarray] = {}
        self._flat_slices_by_k: Dict[int, List[Tuple[int, int]]] = {}
        if not self.entries:
            return
        all_k = set()
        for e in self.entries:
            all_k.update(e.protos.keys())

        for K in sorted(all_k):
            mats = []
            total = 0
            slices: List[Tuple[int, int]] = []
            for e in self.entries:
                p = e.protos.get(K)
                if p is None or p.size == 0:
                    slices.append((total, total))
                    continue
                mats.append(p)
                slices.append((total, total + p.shape[0]))
                total += p.shape[0]
            if mats:
                self._flat_protos[K] = np.concatenate(mats, axis=0)
                self._flat_slices_by_k[K] = slices

        # also collect all video-level means for hubness correction (NNN)
        means = []
        first_vec = None
        for e in self.entries:
            p = e.protos.get(max(e.protos.keys())) if e.protos else None
            if p is not None and p.size > 0:
                if first_vec is None:
                    first_vec = np.zeros_like(p[0])
                means.append(p.mean(axis=0))
            else:
                means.append(None)
        if first_vec is not None:
            means = [m if m is not None else np.zeros_like(first_vec) for m in means]
            self._vid_means = np.stack(means, axis=0)  # (N, D)
        else:
            self._vid_means = None

    # ---- query API --------------------------------------------------

    def search(self,
               query_emb: np.ndarray,
               top_k: int = 10,
               alpha_nnn: float = 0.5,
               tau_qamp: float = 0.01,
               col_beta: float = 0.4,
               topm_rerank: int = 50,
               use_multi_k: bool = True,
               base_scores_context: Optional[np.ndarray] = None,
               ) -> List[Tuple[str, float, float]]:
        """CNPR-style retrieval scoring over the offline index.

        Args:
            query_emb: (D,) text embedding (will be L2-normalized).
            top_k: return length.
            alpha_nnn: fusion weight between NNN z-score and QAMP z-score
                (CNPR best ≈ 0.5).
            tau_qamp: QAMP softmax temperature (CNPR best ≈ 0.01).
            col_beta: column-only softmax temperature for final hubness
                correction (CNPR v6b best ≈ 0.4).
            topm_rerank: only the top-M candidates by base cosine are
                rescored (CNPR default 50).
            use_multi_k: average QAMP across K={2,4,6} prototypes.
            base_scores_context: optional (N,) vector of per-video base
                cosines from context queries, used to estimate NNN
                `vid_mu / vid_std`. If None, we self-bootstrap from the
                current query's base scores alone.

        Returns:
            List[(video_id, final_score, margin)], sorted desc by score.
            margin := final_score[top1] - final_score[top2].
        """
        if not self.entries:
            return []
        q = query_emb.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        N = len(self.entries)

        # ---- 1. base cosine = max over K=6 prototypes (video-level) ------
        # We treat the K=6 protos as the "fine" view, closest to video_embs.
        fine_K = max(self._flat_protos.keys())
        big = self._flat_protos[fine_K]
        slices = self._flat_slices_by_k[fine_K]
        sims = big @ q                               # (sum_K,)
        base = np.full(N, -1e9, dtype=np.float32)
        for i, (s, e) in enumerate(slices):
            if e > s:
                base[i] = float(sims[s:e].max())

        # ---- 2. QAMP score per video (averaged across K if multi-K) ------
        qamp = np.full(N, -1e9, dtype=np.float32)
        k_iter = list(self._flat_protos.keys()) if use_multi_k else [fine_K]
        qamp_acc = np.zeros(N, dtype=np.float32)
        qamp_cnt = 0
        for K in k_iter:
            big_k = self._flat_protos[K]
            sl_k = self._flat_slices_by_k[K]
            sims_k = big_k @ q
            for i, (s, e) in enumerate(sl_k):
                if e > s:
                    seg = sims_k[s:e]
                    z = seg / max(tau_qamp, 1e-6)
                    z = z - z.max()
                    w = np.exp(z)
                    w /= w.sum() + 1e-9
                    qamp_acc[i] += float((w * seg).sum())
            qamp_cnt += 1
        qamp = qamp_acc / max(qamp_cnt, 1)

        # ---- 3. NNN z-score ---------------------------------------------
        if base_scores_context is not None and base_scores_context.ndim == 2 \
                and base_scores_context.shape[1] == N:
            # Full (n_queries, N_videos) context — use column stats (CNPR).
            vid_mu = base_scores_context.mean(axis=0).astype(np.float32)
            vid_std = (base_scores_context.std(axis=0) + 1e-8).astype(np.float32)
        else:
            # Fallback: per-video mean-proto similarity as a proxy for mu.
            if self._vid_means is not None:
                vid_mu = (self._vid_means @ q).astype(np.float32)
            else:
                vid_mu = np.full(N, float(base.mean()), dtype=np.float32)
            vid_std = np.full(N, float(base.std() + 1e-8), dtype=np.float32)

        # ---- 4. Rerank top-M by fused NNN + QAMP ------------------------
        m = min(topm_rerank, N)
        cand = np.argpartition(-base, m - 1)[:m]

        nnn_vals = (base[cand] - vid_mu[cand]) / vid_std[cand]
        qamp_vals = qamp[cand]
        nnn_z = (nnn_vals - nnn_vals.mean()) / (nnn_vals.std() + 1e-8)
        qamp_z = (qamp_vals - qamp_vals.mean()) / (qamp_vals.std() + 1e-8)
        fused = (1.0 - alpha_nnn) * nnn_z + alpha_nnn * qamp_z

        # keep non-candidates below the lowest fused rerank score
        fused_shifted = fused - fused.min() + float(np.partition(base, -m)[-m - 1]
                                                     if N > m else 0.0) + 1e-6
        scores = base.astype(np.float32).copy()
        for ci, j in enumerate(cand):
            scores[j] = fused_shifted[ci]

        # ---- 5. Column-only softmax normalization (CNPR v6b) ------------
        # NOTE: true col-only softmax normalizes each column (video)
        # across ALL queries. For a single-query search we approximate it
        # by softmaxing across videos in this row. For batch eval that
        # needs the proper column statistics, use `search_batch()`.
        if col_beta > 0:
            z = scores / max(col_beta, 1e-6)
            z = z - z.max()
            e = np.exp(z)
            scores = e / (e.sum() + 1e-9)

        # ---- 6. Top-K + margin ------------------------------------------
        order = np.argsort(-scores)
        top1 = float(scores[order[0]])
        top2 = float(scores[order[1]]) if len(order) > 1 else 0.0
        margin = top1 - top2

        out = []
        for i in order[:top_k]:
            out.append((self.entries[i].video_id, float(scores[i]), margin))
        return out

    # ---- misc --------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.entries)

    def bytes_on_disk(self) -> int:
        return sum(e.payload_bytes() for e in self.entries)

    def summary(self) -> Dict:
        return {
            "n_videos": self.size,
            "avg_keyframes":
                float(np.mean([len(e.key_ts) for e in self.entries]))
                if self.entries else 0.0,
            "total_bytes": self.bytes_on_disk(),
            "K_values": list(next(iter(self.entries)).protos.keys())
                if self.entries else [],
        }

    # ==================================================================
    #  Phase-3: metadata-aware retrieval
    # ==================================================================

    def search_batch_with_meta(self,
                                query_embs: np.ndarray,
                                intents: List[QueryIntent],
                                top_k: int = 10,
                                alpha_nnn: float = 0.7,
                                tau_qamp: float = 0.10,
                                col_beta: float = 0.4,
                                topm_rerank: int = 500,
                                meta_filter: Optional[MetaFilter] = None,
                                meta_alpha: float = 1.0,
                                use_hard_filter: bool = True,
                                use_meta_fusion: bool = False,
                                col_softmax_after_filter: bool = True,
                                ) -> List[List[Tuple[str, float, float, Dict]]]:
        """Batch hybrid retrieval with corrected col-softmax + meta-fusion.

        Important design fixes (informed by ablation):
          * `col_softmax_after_filter=True` (default) — the col-softmax
            normalisation is applied to the SURVIVING candidates only,
            after the hard meta-filter removes irrelevant videos. This
            preserves the semantic signal for the remaining videos
            instead of diluting it across 1000 entries.
          * `use_meta_fusion=False` (default) — soft α·sem+(1-α)·meta
            blending is OFF by default because in practice the soft
            metadata score lifts many irrelevant videos that match the
            same coarse tag (e.g. all "coast" videos), drowning out the
            semantic top-1. The hard filter alone gives the best
            precision; soft fusion is kept as an opt-in for cases where
            the user query has weak semantic signal.

        Pipeline per query:
          1. Pre-filter pass: hard mask from MetaFilter.
          2. Compute base + QAMP + NNN scores (vectorised, no col-softmax).
          3. Restrict to surviving candidates.
          4. Apply col-softmax (axis=0) on the masked score matrix.
          5. Top-M rerank within the masked set.
          6. (Optional) per-query soft fusion with meta_alpha.
        """
        if not self.entries:
            return [[] for _ in range(query_embs.shape[0])]
        mf = meta_filter or MetaFilter()
        metas = [e.metadata for e in self.entries]
        Nq = query_embs.shape[0]
        Nv = len(self.entries)

        # --- Step 1: build per-query hard masks ---
        masks = np.ones((Nq, Nv), dtype=bool)
        if use_hard_filter:
            for q_idx in range(Nq):
                it = intents[q_idx]
                if it.has_constraint():
                    masks[q_idx] = mf.filter(metas, it).mask

        # --- Step 2: get raw semantic scores (no col-softmax yet) ---
        all_hits = self.search_batch(
            query_embs, top_k=Nv,
            alpha_nnn=alpha_nnn, tau_qamp=tau_qamp,
            col_beta=0.0,                      # no col-softmax now
            topm_rerank=topm_rerank,
        )
        id_to_idx = {e.video_id: i for i, e in enumerate(self.entries)}
        sem_scores = np.zeros((Nq, Nv), dtype=np.float32)
        for q_idx, hits in enumerate(all_hits):
            for vid, sc, _m in hits:
                sem_scores[q_idx, id_to_idx[vid]] = sc

        # --- Step 3: mask out filtered videos (set to -inf for col-softmax) ---
        masked = np.where(masks, sem_scores, -1e9)

        # --- Step 4: col-softmax on the *masked* scores (axis=0) ---
        if col_beta > 0 and col_softmax_after_filter:
            # col-max ignoring -inf entries (where mask is False)
            col_max = masked.max(axis=0, keepdims=True)
            z = (masked - col_max) / max(col_beta, 1e-6)
            # exp of -inf differences → 0; safe.
            e = np.exp(z)
            denom = e.sum(axis=0, keepdims=True) + 1e-12
            sem_norm = e / denom
            # for queries with no surviving candidates, fall back
            sem_norm = np.where(masks, sem_norm, -1e9).astype(np.float32)
        elif col_beta > 0 and not col_softmax_after_filter:
            # legacy global col-softmax then mask
            col_max = sem_scores.max(axis=0, keepdims=True)
            z = (sem_scores - col_max) / max(col_beta, 1e-6)
            e = np.exp(z)
            sem_norm = e / (e.sum(axis=0, keepdims=True) + 1e-12)
            sem_norm = np.where(masks, sem_norm, -1e9).astype(np.float32)
        else:
            sem_norm = masked

        # --- Step 5+6: optional soft fusion + final top-K ---
        results: List[List[Tuple[str, float, float, Dict]]] = []
        for q_idx in range(Nq):
            it = intents[q_idx]
            sem_q = sem_norm[q_idx].copy()

            if use_meta_fusion and it.has_constraint() and meta_alpha < 1.0:
                meta_soft = mf.soft_score(metas, it)
                fused = meta_alpha * sem_q + (1.0 - meta_alpha) * meta_soft
                fused = np.where(masks[q_idx], fused, -1e9)
            else:
                fused = sem_q

            order = np.argsort(-fused)
            top1 = float(fused[order[0]])
            top2 = float(fused[order[1]]) if len(order) > 1 else 0.0
            margin = top1 - top2

            row = []
            for i in order[:top_k]:
                debug = {
                    "semantic": float(sem_scores[q_idx, i]),
                    "fused":    float(fused[i]),
                    "passed_filter": bool(masks[q_idx, i]),
                }
                row.append((self.entries[i].video_id, float(fused[i]),
                             margin, debug))
            results.append(row)
        return results

    def search_with_meta(self,
                         query_emb: np.ndarray,
                         intent: QueryIntent,
                         top_k: int = 10,
                         alpha_nnn: float = 0.7,
                         tau_qamp: float = 0.10,
                         col_beta: float = 0.4,
                         topm_rerank: int = 500,
                         meta_filter: Optional[MetaFilter] = None,
                         meta_alpha: float = 1.0,
                         use_hard_filter: bool = True,
                         use_meta_fusion: bool = False,
                         col_softmax_after_filter: bool = True,
                         ) -> List[Tuple[str, float, float, Dict]]:
        """Single-query hybrid retrieval (delegates to batch path)."""
        rows = self.search_batch_with_meta(
            np.expand_dims(query_emb, 0), [intent], top_k=top_k,
            alpha_nnn=alpha_nnn, tau_qamp=tau_qamp, col_beta=col_beta,
            topm_rerank=topm_rerank, meta_filter=meta_filter,
            meta_alpha=meta_alpha, use_hard_filter=use_hard_filter,
            use_meta_fusion=use_meta_fusion,
            col_softmax_after_filter=col_softmax_after_filter,
        )
        return rows[0]

    def search_batch(self,
                     query_embs: np.ndarray,
                     top_k: int = 10,
                     alpha_nnn: float = 0.7,
                     tau_qamp: float = 0.10,
                     col_beta: float = 0.4,
                     topm_rerank: int = 500,
                     use_multi_k: bool = True,
                     ) -> List[List[Tuple[str, float, float]]]:
        """Evaluate a matrix of queries at once with correct col-softmax.

        Defaults are the joint-optimum from the MSR-VTT 1K hyperparam
        ablation (J_a0.7_t0.1_c0.4_m500): R@1=69.50%, ms/q=2.54.
        """
        if not self.entries:
            return [[] for _ in range(query_embs.shape[0])]

        Q = query_embs.astype(np.float32)
        Q = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-9)
        Nq = Q.shape[0]
        Nv = len(self.entries)

        # --- 1. base scores (max over K=6 protos) ---
        fine_K = max(self._flat_protos.keys())
        big = self._flat_protos[fine_K]
        sl = self._flat_slices_by_k[fine_K]
        sims_all = Q @ big.T                                # (Nq, sum_K)
        base = np.full((Nq, Nv), -1e9, dtype=np.float32)
        for j, (s, e) in enumerate(sl):
            if e > s:
                base[:, j] = sims_all[:, s:e].max(axis=1)

        # --- 2. QAMP scores per K, averaged if multi-K ---
        qamp_accum = np.zeros((Nq, Nv), dtype=np.float32)
        k_iter = list(self._flat_protos.keys()) if use_multi_k else [fine_K]
        for K in k_iter:
            big_k = self._flat_protos[K]
            sl_k = self._flat_slices_by_k[K]
            sims_k = Q @ big_k.T                            # (Nq, sum_K)
            for j, (s, e) in enumerate(sl_k):
                if e > s:
                    seg = sims_k[:, s:e]                    # (Nq, k)
                    z = seg / max(tau_qamp, 1e-6)
                    z = z - z.max(axis=-1, keepdims=True)
                    w = np.exp(z)
                    w /= w.sum(axis=-1, keepdims=True) + 1e-9
                    qamp_accum[:, j] += (w * seg).sum(axis=-1)
        qamp = qamp_accum / len(k_iter)

        # --- 3. NNN z-score using column stats (CNPR exact) ---
        vid_mu = base.mean(axis=0)
        vid_std = base.std(axis=0) + 1e-8

        # --- 4. per-row top-M rerank ---
        out_scores = base.copy()
        m = min(topm_rerank, Nv)
        for i in range(Nq):
            row = base[i]
            cand = np.argpartition(-row, m - 1)[:m]
            nnn_vals = (row[cand] - vid_mu[cand]) / vid_std[cand]
            qa_vals = qamp[i, cand]
            nnn_z = (nnn_vals - nnn_vals.mean()) / (nnn_vals.std() + 1e-8)
            qa_z = (qa_vals - qa_vals.mean()) / (qa_vals.std() + 1e-8)
            fused = (1.0 - alpha_nnn) * nnn_z + alpha_nnn * qa_z
            non_max = float(np.partition(-row, m)[m]) if Nv > m else 0.0
            fused = fused - fused.min() + non_max + 1e-6
            for ci, j in enumerate(cand):
                out_scores[i, j] = fused[ci]

        # --- 5. CORRECT col-only softmax (CNPR v6b) ---
        #    softmax along axis=0: per video, across queries
        if col_beta > 0:
            col_max = out_scores.max(axis=0, keepdims=True)
            z = (out_scores - col_max) / max(col_beta, 1e-6)
            e = np.exp(z)
            out_scores = e / (e.sum(axis=0, keepdims=True) + 1e-12)

        # --- 6. rank + margin ---
        results: List[List[Tuple[str, float, float]]] = []
        order = np.argsort(-out_scores, axis=1)
        for i in range(Nq):
            top1 = float(out_scores[i, order[i, 0]])
            top2 = float(out_scores[i, order[i, 1]]) if Nv > 1 else 0.0
            margin = top1 - top2
            rows = [
                (self.entries[int(order[i, k])].video_id,
                 float(out_scores[i, order[i, k]]),
                 margin)
                for k in range(min(top_k, Nv))
            ]
            results.append(rows)
        return results
