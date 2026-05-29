"""Shared schemas for the CSER implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ExpertSpec:
    """Static metadata for one frozen expert."""

    expert_id: str
    cost: float = 1.0
    mandatory: bool = False
    output_kind: str = "score"
    description: str = ""


@dataclass
class ExpertScore:
    """Per-video output produced by a frozen expert for one query."""

    scores: np.ndarray
    keep_mask: Optional[np.ndarray] = None
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if self.keep_mask is not None:
            self.keep_mask = np.asarray(self.keep_mask, dtype=bool).reshape(-1)
            if self.keep_mask.shape[0] != self.scores.shape[0]:
                raise ValueError("keep_mask length must match scores length")


@dataclass
class CSERSubsetResult:
    """Execution result for a selected expert subset."""

    selected_experts: Tuple[str, ...]
    rank: int
    recall_at: Dict[int, int]
    mrr: float
    gt_filtered: bool
    cost: float
    candidate_count: int
    scores: np.ndarray = field(repr=False)
    keep_mask: np.ndarray = field(repr=False)

    def to_dict(self) -> Dict[str, object]:
        return {
            "selected_experts": list(self.selected_experts),
            "rank": int(self.rank),
            "recall_at": {str(k): int(v) for k, v in self.recall_at.items()},
            "mrr": float(self.mrr),
            "gt_filtered": bool(self.gt_filtered),
            "cost": float(self.cost),
            "candidate_count": int(self.candidate_count),
        }


@dataclass
class CSERDecision:
    """Greedy selector decision trace."""

    selected_experts: Tuple[str, ...]
    budget_used: float
    conformal_set_size: int
    step_values: List[Dict[str, object]] = field(default_factory=list)
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "selected_experts": list(self.selected_experts),
            "budget_used": float(self.budget_used),
            "conformal_set_size": int(self.conformal_set_size),
            "step_values": list(self.step_values),
            "used_fallback": bool(self.used_fallback),
        }


DEFAULT_EXPERTS: Tuple[ExpertSpec, ...] = (
    ExpertSpec(
        expert_id="clip_semantic",
        cost=1.0,
        mandatory=True,
        output_kind="dense_score",
        description="MobileCLIP semantic retrieval expert",
    ),
    ExpertSpec(
        expert_id="face_detect",
        cost=1.0,
        mandatory=False,
        output_kind="score_mask",
        description="SCRFD face detection expert",
    ),
    ExpertSpec(
        expert_id="arcface",
        cost=1.0,
        mandatory=False,
        output_kind="dense_score",
        description="ArcFace identity matching expert",
    ),
    ExpertSpec(
        expert_id="highlight",
        cost=1.0,
        mandatory=False,
        output_kind="dense_score",
        description="MomentDETR highlight expert",
    ),
    ExpertSpec(
        expert_id="scene",
        cost=1.0,
        mandatory=False,
        output_kind="score_mask",
        description="MobileNetV3 scene expert",
    ),
)

DEFAULT_EXPERT_IDS: Tuple[str, ...] = tuple(e.expert_id for e in DEFAULT_EXPERTS)


def expert_id_to_index(
    expert_specs: Sequence[ExpertSpec] = DEFAULT_EXPERTS,
) -> Dict[str, int]:
    return {spec.expert_id: i for i, spec in enumerate(expert_specs)}


def mandatory_expert_ids(
    expert_specs: Sequence[ExpertSpec] = DEFAULT_EXPERTS,
) -> Tuple[str, ...]:
    return tuple(spec.expert_id for spec in expert_specs if spec.mandatory)


def expert_cost(
    expert_id: str,
    expert_specs: Sequence[ExpertSpec] = DEFAULT_EXPERTS,
) -> float:
    for spec in expert_specs:
        if spec.expert_id == expert_id:
            return float(spec.cost)
    raise KeyError(f"Unknown expert_id: {expert_id}")
