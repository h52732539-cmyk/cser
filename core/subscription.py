"""Task subscription descriptor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskSubscription:
    """Describes how a task wants to consume frames."""

    task_id: str

    # Target fps for each stage
    sparse_fps: float = 0.5
    dense_fps: float = 2.0

    # Minimum gap between two sampled frames for this task (seconds)
    min_gap_sec: float = 0.1

    # Budget & priority for conflict resolution
    priority: int = 5
    max_frames_sparse: int = 100
    max_frames_dense: int = 200

    # Behavioral flags
    can_produce_interest: bool = False
    """If True the task is allowed to emit InterestSignal in Stage 1."""

    gated_by: Optional[str] = None
    """If set, this task only runs in Stage 2 and only on frames where the
    gating task was positive (e.g. face_emb gated_by face_det)."""

    respects_metadata: bool = True
    """If False the task ignores the prefilter mask (always samples)."""
