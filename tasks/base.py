"""Abstract base class for all tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from core.subscription import TaskSubscription
from core.types import Frame, InterestSignal, TaskResult


class BaseTask(ABC):
    """Base class all tasks must inherit from.

    A task may implement either or both of the two stages. If
    `sub.gated_by` is set, `process_sparse` will not be called by the
    framework; the task is only invoked in Stage 2 on frames where the
    gating task was positive.
    """

    def __init__(self, subscription: TaskSubscription) -> None:
        self.sub = subscription
        self.task_id = subscription.task_id

    @abstractmethod
    def process_sparse(self, frames: List[Frame]) -> Optional[InterestSignal]:
        """Stage 1: lightweight processing, may emit an InterestSignal."""
        raise NotImplementedError

    @abstractmethod
    def process_dense(self, frames: List[Frame]) -> None:
        """Stage 2: full-featured processing on interest regions."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> TaskResult:
        """Return the final TaskResult."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-video state. Default no-op."""
        pass
