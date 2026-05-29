"""Scene classification task (sparse-only, no Stage 2 needed)."""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from core.types import Frame, InterestSignal, TaskResult
from .base import BaseTask


class SceneClassificationTask(BaseTask):
    """Per-frame scene label; produces a sequence of scene labels."""

    def __init__(self, subscription, classifier) -> None:
        super().__init__(subscription)
        self.classifier = classifier
        self.reset()

    def reset(self) -> None:
        self._labels: List[tuple] = []  # (timestamp, label)

    def process_sparse(self, frames: List[Frame]) -> Optional[InterestSignal]:
        if not frames:
            return None
        labels = self.classifier.classify([f.image for f in frames])
        for f, lab in zip(frames, labels):
            self._labels.append((float(f.timestamp), lab))
        return None  # no interest emitted

    def process_dense(self, frames: List[Frame]) -> None:
        if not frames:
            return
        labels = self.classifier.classify([f.image for f in frames])
        for f, lab in zip(frames, labels):
            self._labels.append((float(f.timestamp), lab))

    def finalize(self) -> TaskResult:
        self._labels.sort(key=lambda x: x[0])
        histogram = Counter(lab for _, lab in self._labels)
        dominant = (
            histogram.most_common(1)[0][0] if histogram else "unknown"
        )
        return TaskResult(
            task_id=self.task_id,
            payload={
                "labels": self._labels,
                "histogram": dict(histogram),
                "dominant": dominant,
            },
            metrics={
                "n_frames": float(len(self._labels)),
                "n_classes": float(len(histogram)),
            },
        )
