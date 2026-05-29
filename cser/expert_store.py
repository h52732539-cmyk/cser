"""Cached frozen-expert outputs for CSER.

The store is deliberately lightweight: it holds gallery-side expert outputs
and converts them into per-query score vectors. Real heavy model extraction can
be added behind the same cache format without changing the planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .schema import DEFAULT_EXPERT_IDS, ExpertScore


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=axis, keepdims=True) + 1e-9)


@dataclass
class ExpertOutputStore:
    """Gallery-side outputs for all CSER experts."""

    video_ids: List[str]
    clip_video_embs: np.ndarray
    face_presence: np.ndarray
    face_embs: np.ndarray
    highlight_scores: np.ndarray
    scene_labels: np.ndarray
    scene_vocab: List[str]

    def __post_init__(self) -> None:
        n = len(self.video_ids)
        self.clip_video_embs = _l2_normalize(self.clip_video_embs)
        self.face_presence = np.asarray(self.face_presence, dtype=np.float32).reshape(n)
        self.face_embs = _l2_normalize(self.face_embs)
        self.highlight_scores = np.asarray(self.highlight_scores, dtype=np.float32).reshape(n)
        self.scene_labels = np.asarray(self.scene_labels).astype(str).reshape(n)
        self._id_to_idx = {vid: i for i, vid in enumerate(self.video_ids)}

    @property
    def size(self) -> int:
        return len(self.video_ids)

    @property
    def dim(self) -> int:
        return int(self.clip_video_embs.shape[1])

    def index_of(self, video_id: str) -> int:
        return self._id_to_idx[video_id]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            video_ids=np.asarray(self.video_ids),
            clip_video_embs=self.clip_video_embs.astype(np.float32),
            face_presence=self.face_presence.astype(np.float32),
            face_embs=self.face_embs.astype(np.float32),
            highlight_scores=self.highlight_scores.astype(np.float32),
            scene_labels=self.scene_labels.astype(str),
            scene_vocab=np.asarray(self.scene_vocab),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExpertOutputStore":
        data = np.load(path, allow_pickle=True)
        return cls(
            video_ids=[str(x) for x in data["video_ids"]],
            clip_video_embs=data["clip_video_embs"].astype(np.float32),
            face_presence=data["face_presence"].astype(np.float32),
            face_embs=data["face_embs"].astype(np.float32),
            highlight_scores=data["highlight_scores"].astype(np.float32),
            scene_labels=data["scene_labels"].astype(str),
            scene_vocab=[str(x) for x in data["scene_vocab"]],
        )

    @classmethod
    def synthetic(
        cls,
        n_videos: int = 64,
        dim: int = 512,
        face_dim: int = 128,
        seed: int = 42,
    ) -> "ExpertOutputStore":
        """Build deterministic mock gallery outputs for tests and smoke runs."""
        rng = np.random.default_rng(seed)
        video_ids = [f"vid_{i:04d}" for i in range(n_videos)]
        clip = _l2_normalize(rng.normal(size=(n_videos, dim)).astype(np.float32))

        face_presence = (rng.random(n_videos) > 0.45).astype(np.float32)
        face_embs = _l2_normalize(rng.normal(size=(n_videos, face_dim)).astype(np.float32))
        face_embs *= face_presence[:, None]

        highlight_scores = rng.beta(2.0, 4.0, size=n_videos).astype(np.float32)
        scene_vocab = ["indoor", "outdoor", "nature", "urban", "sport", "party"]
        scene_labels = rng.choice(scene_vocab, size=n_videos)

        return cls(
            video_ids=video_ids,
            clip_video_embs=clip,
            face_presence=face_presence,
            face_embs=face_embs,
            highlight_scores=highlight_scores,
            scene_labels=scene_labels,
            scene_vocab=scene_vocab,
        )

    @classmethod
    def from_msrvtt_cache(cls, cache_npz: str | Path, seed: int = 42) -> "ExpertOutputStore":
        """Create a CSER cache from the existing MSR-VTT prototype archive.

        The cache contains real semantic prototypes when available. Non-semantic
        experts are initialized as deterministic pseudo outputs, so the strict
        CSER pipeline remains runnable before heavyweight expert extraction is
        wired into a local machine.
        """
        rng = np.random.default_rng(seed)
        data = np.load(cache_npz, allow_pickle=True)
        video_ids = [str(x) for x in data["video_ids"]]
        protos = data["protos"].astype(np.float32)
        if protos.ndim == 3:
            clip = protos.mean(axis=1)
        else:
            clip = protos.reshape(len(video_ids), -1)
        clip = _l2_normalize(clip)

        n = len(video_ids)
        face_presence = (rng.random(n) > 0.55).astype(np.float32)
        face_embs = _l2_normalize(rng.normal(size=(n, 128)).astype(np.float32))
        face_embs *= face_presence[:, None]
        highlight_scores = rng.beta(2.0, 5.0, size=n).astype(np.float32)
        scene_vocab = ["indoor", "outdoor", "nature", "urban", "sport", "party"]
        scene_labels = rng.choice(scene_vocab, size=n)
        return cls(
            video_ids=video_ids,
            clip_video_embs=clip,
            face_presence=face_presence,
            face_embs=face_embs,
            highlight_scores=highlight_scores,
            scene_labels=scene_labels,
            scene_vocab=scene_vocab,
        )

    def query_context_for_gt(
        self,
        gt_video_id: str,
        rng: Optional[np.random.Generator] = None,
        include_filters: bool = True,
    ) -> Dict[str, object]:
        """Create a plausible query context from a known GT video."""
        rng = rng or np.random.default_rng(0)
        idx = self.index_of(gt_video_id)
        has_face = bool(self.face_presence[idx] > 0.5)
        ctx: Dict[str, object] = {
            "requires_face": bool(include_filters and has_face and rng.random() < 0.55),
            "target_face_embedding": self.face_embs[idx].copy() if has_face else None,
            "wants_highlight": bool(rng.random() < 0.45),
            "scene_label": str(self.scene_labels[idx]) if rng.random() < 0.65 else None,
            "requires_scene_filter": bool(include_filters and rng.random() < 0.55),
        }
        return ctx

    def score_expert(
        self,
        expert_id: str,
        query_emb: np.ndarray,
        query_context: Optional[Mapping[str, object]] = None,
    ) -> ExpertScore:
        query_context = query_context or {}
        if expert_id not in DEFAULT_EXPERT_IDS:
            raise KeyError(f"Unknown CSER expert: {expert_id}")

        if expert_id == "clip_semantic":
            q = _l2_normalize(np.asarray(query_emb, dtype=np.float32).reshape(1, -1))[0]
            return ExpertScore(scores=self.clip_video_embs @ q)

        if expert_id == "face_detect":
            scores = self.face_presence.astype(np.float32)
            keep = None
            if bool(query_context.get("requires_face", False)):
                keep = scores > 0.5
            return ExpertScore(scores=scores, keep_mask=keep)

        if expert_id == "arcface":
            target = query_context.get("target_face_embedding")
            if target is None:
                return ExpertScore(scores=np.zeros(self.size, dtype=np.float32))
            t = _l2_normalize(np.asarray(target, dtype=np.float32).reshape(1, -1))[0]
            scores = self.face_embs @ t
            threshold = float(query_context.get("face_threshold", 0.25))
            keep = scores >= threshold if bool(query_context.get("requires_face", False)) else None
            return ExpertScore(scores=scores.astype(np.float32), keep_mask=keep)

        if expert_id == "highlight":
            if bool(query_context.get("wants_highlight", False)):
                scores = self.highlight_scores
            else:
                scores = np.zeros(self.size, dtype=np.float32)
            return ExpertScore(scores=scores)

        scene_label = query_context.get("scene_label")
        if expert_id == "scene":
            if scene_label is None:
                return ExpertScore(scores=np.zeros(self.size, dtype=np.float32))
            matches = self.scene_labels == str(scene_label)
            scores = np.where(matches, 1.0, 0.15).astype(np.float32)
            keep = matches if bool(query_context.get("requires_scene_filter", True)) else None
            return ExpertScore(scores=scores, keep_mask=keep)

        raise KeyError(expert_id)

    def score_all(
        self,
        query_emb: np.ndarray,
        query_context: Optional[Mapping[str, object]] = None,
        expert_ids: Sequence[str] = DEFAULT_EXPERT_IDS,
    ) -> Dict[str, ExpertScore]:
        return {
            expert_id: self.score_expert(expert_id, query_emb, query_context)
            for expert_id in expert_ids
        }
