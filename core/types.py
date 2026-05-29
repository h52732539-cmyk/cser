"""Core data types for the multi-model framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Set

import numpy as np


class SamplingStage(Enum):
    SPARSE = "sparse"  # Stage 1: preview
    DENSE = "dense"    # Stage 2: refine


@dataclass
class Frame:
    """Unified frame representation passed to tasks."""
    video_id: str
    frame_idx: int
    timestamp: float
    image: np.ndarray  # HxWx3 uint8 RGB
    stage: SamplingStage


@dataclass
class Interval:
    """Temporal interval for interest regions / dense sampling windows."""
    start: float
    end: float
    score: float = 0.0
    source_task: str = ""

    def length(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class InterestSignal:
    """Emitted by Stage-1 tasks to request dense refinement regions."""
    task_id: str
    intervals: List[Interval] = field(default_factory=list)


@dataclass
class TaskResult:
    """Final output of a task after both stages."""
    task_id: str
    payload: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FrameRequest:
    """Sampling request produced by the scheduler."""
    video_id: str
    frame_idx: int
    timestamp: float
    stage: SamplingStage
    subscribers: Set[str] = field(default_factory=set)
