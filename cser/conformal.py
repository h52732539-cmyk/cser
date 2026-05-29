"""Split and Mondrian conformal safety gates for CSER."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .subset_executor import normalize_scores


def nonconformity_from_scores(scores: np.ndarray) -> np.ndarray:
    return 1.0 - normalize_scores(scores)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    arr = np.sort(np.asarray(scores, dtype=np.float32).reshape(-1))
    if arr.size == 0:
        return float("inf")
    rank = int(np.ceil((arr.size + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), arr.size - 1)
    return float(arr[rank])


def difficulty_from_scores(scores: np.ndarray) -> float:
    norm = normalize_scores(scores)
    order = np.sort(norm)[::-1]
    top1 = float(order[0]) if order.size else 0.0
    top2 = float(order[1]) if order.size > 1 else 0.0
    margin = top1 - top2
    probs = norm / (float(norm.sum()) + 1e-9)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return float(entropy - margin)


def gt_indices_from_ids(video_ids: Sequence[str], gt_video_ids: Sequence[str]) -> np.ndarray:
    lookup = {vid: i for i, vid in enumerate(video_ids)}
    return np.asarray([lookup[g] for g in gt_video_ids], dtype=np.int32)


@dataclass
class SplitConformalCalibrator:
    alpha: float = 0.05
    threshold: float = float("inf")
    n_calibration: int = 0

    def fit(self, score_matrix: np.ndarray, gt_indices: Sequence[int]) -> "SplitConformalCalibrator":
        scores = np.asarray(score_matrix, dtype=np.float32)
        gt = np.asarray(gt_indices, dtype=np.int32)
        cal_scores = []
        for i, g in enumerate(gt):
            cal_scores.append(float(nonconformity_from_scores(scores[i])[g]))
        self.threshold = conformal_quantile(np.asarray(cal_scores, dtype=np.float32), self.alpha)
        self.n_calibration = int(len(cal_scores))
        return self

    def predict(self, scores: np.ndarray, difficulty: Optional[float] = None) -> np.ndarray:
        nc = nonconformity_from_scores(scores)
        return nc <= self.threshold

    def coverage(self, score_matrix: np.ndarray, gt_indices: Sequence[int]) -> float:
        scores = np.asarray(score_matrix, dtype=np.float32)
        gt = np.asarray(gt_indices, dtype=np.int32)
        hits = [bool(self.predict(scores[i])[g]) for i, g in enumerate(gt)]
        return float(np.mean(hits)) if hits else 0.0

    def report(self, score_matrix: np.ndarray, gt_indices: Sequence[int]) -> Dict[str, float]:
        sizes = [int(self.predict(row).sum()) for row in score_matrix]
        return {
            "alpha": float(self.alpha),
            "threshold": float(self.threshold),
            "n_calibration": float(self.n_calibration),
            "coverage": float(self.coverage(score_matrix, gt_indices)),
            "avg_set_size": float(np.mean(sizes)) if sizes else 0.0,
        }


@dataclass
class MondrianConformalCalibrator:
    alpha: float = 0.05
    n_bins: int = 3
    min_bin_size: int = 30
    global_calibrator: SplitConformalCalibrator = field(default_factory=SplitConformalCalibrator)
    bin_edges: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float32))
    thresholds: Dict[int, float] = field(default_factory=dict)
    bin_counts: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.global_calibrator.alpha = self.alpha

    def fit(
        self,
        score_matrix: np.ndarray,
        gt_indices: Sequence[int],
        difficulties: Optional[Sequence[float]] = None,
    ) -> "MondrianConformalCalibrator":
        scores = np.asarray(score_matrix, dtype=np.float32)
        gt = np.asarray(gt_indices, dtype=np.int32)
        if difficulties is None:
            diffs = np.asarray([difficulty_from_scores(row) for row in scores], dtype=np.float32)
        else:
            diffs = np.asarray(difficulties, dtype=np.float32)

        self.global_calibrator = SplitConformalCalibrator(alpha=self.alpha).fit(scores, gt)
        if scores.shape[0] == 0:
            return self

        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.bin_edges = np.quantile(diffs, quantiles).astype(np.float32) if quantiles.size else np.asarray([])
        bins = np.digitize(diffs, self.bin_edges, right=True)

        self.thresholds = {}
        self.bin_counts = {}
        for b in range(self.n_bins):
            idx = np.where(bins == b)[0]
            self.bin_counts[b] = int(idx.size)
            if idx.size < self.min_bin_size:
                self.thresholds[b] = float(self.global_calibrator.threshold)
                continue
            cal_scores = [
                float(nonconformity_from_scores(scores[i])[gt[i]])
                for i in idx
            ]
            self.thresholds[b] = conformal_quantile(np.asarray(cal_scores), self.alpha)
        return self

    def _bin_for(self, difficulty: float) -> int:
        if self.bin_edges.size == 0:
            return 0
        return int(np.digitize([difficulty], self.bin_edges, right=True)[0])

    def predict(self, scores: np.ndarray, difficulty: Optional[float] = None) -> np.ndarray:
        diff = difficulty_from_scores(scores) if difficulty is None else float(difficulty)
        b = self._bin_for(diff)
        threshold = self.thresholds.get(b, self.global_calibrator.threshold)
        return nonconformity_from_scores(scores) <= threshold

    def coverage(
        self,
        score_matrix: np.ndarray,
        gt_indices: Sequence[int],
        difficulties: Optional[Sequence[float]] = None,
    ) -> float:
        scores = np.asarray(score_matrix, dtype=np.float32)
        gt = np.asarray(gt_indices, dtype=np.int32)
        if difficulties is None:
            difficulties = [difficulty_from_scores(row) for row in scores]
        hits = [
            bool(self.predict(scores[i], float(difficulties[i]))[gt[i]])
            for i in range(len(gt))
        ]
        return float(np.mean(hits)) if hits else 0.0

    def report(
        self,
        score_matrix: np.ndarray,
        gt_indices: Sequence[int],
        difficulties: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        scores = np.asarray(score_matrix, dtype=np.float32)
        if difficulties is None:
            difficulties = [difficulty_from_scores(row) for row in scores]
        sizes = [
            int(self.predict(scores[i], float(difficulties[i])).sum())
            for i in range(scores.shape[0])
        ]
        return {
            "alpha": float(self.alpha),
            "global_threshold": float(self.global_calibrator.threshold),
            "thresholds": {str(k): float(v) for k, v in self.thresholds.items()},
            "bin_counts": {str(k): int(v) for k, v in self.bin_counts.items()},
            "coverage": float(self.coverage(scores, gt_indices, difficulties)),
            "avg_set_size": float(np.mean(sizes)) if sizes else 0.0,
        }
