"""Route schema — defines the structure of a retrieval route.

A RetrievalRoute specifies:
  - Which metadata axes to hard-filter (remove non-matching videos)
  - Which metadata axes to soft-rerank (boost matching videos)
  - Candidate pool size (topm after initial retrieval)
  - Reranking strategy (none / qamp / nnn_qamp / col_softmax_post_filter)
  - Budget tier (controls whether dense refinement / image model calls allowed)

Validation rules:
  - 'semantic' is never a hard/soft axis (it's the base retrieval)
  - hard_axes and soft_axes must not overlap
  - hard_axes can only be: time, geo, motion, device
  - candidate_topm ∈ {100, 300, 500, 1000}
  - low budget routes cannot allow_dense_refinement
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

Axis = Literal["time", "geo", "motion", "device"]
RerankMode = Literal["none", "qamp", "nnn_qamp", "col_softmax_post_filter"]
BudgetTier = Literal["low", "medium", "high", "full"]

VALID_AXES = ("time", "geo", "motion", "device")
VALID_TOPM = (100, 300, 500, 1000)
VALID_RERANK = ("none", "qamp", "nnn_qamp", "col_softmax_post_filter")
VALID_BUDGET = ("low", "medium", "high", "full")


@dataclass(frozen=True)
class RetrievalRoute:
    route_id: str
    description: str = ""
    hard_axes: Tuple[str, ...] = ()
    soft_axes: Tuple[str, ...] = ()
    candidate_topm: int = 500
    rerank_mode: str = "nnn_qamp"
    budget_tier: str = "low"
    use_offline_index: bool = True
    allow_image_model_calls: bool = False
    allow_dense_refinement: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        for a in self.hard_axes:
            if a not in VALID_AXES:
                raise ValueError(
                    f"hard_axes contains invalid axis '{a}'. "
                    f"Must be one of {VALID_AXES}."
                )
        for a in self.soft_axes:
            if a not in VALID_AXES:
                raise ValueError(
                    f"soft_axes contains invalid axis '{a}'. "
                    f"Must be one of {VALID_AXES}."
                )
        overlap = set(self.hard_axes) & set(self.soft_axes)
        if overlap:
            raise ValueError(
                f"hard_axes and soft_axes must not overlap. "
                f"Overlap: {overlap}"
            )
        if self.candidate_topm not in VALID_TOPM:
            raise ValueError(
                f"candidate_topm={self.candidate_topm} not in {VALID_TOPM}"
            )
        if self.rerank_mode not in VALID_RERANK:
            raise ValueError(
                f"rerank_mode='{self.rerank_mode}' not in {VALID_RERANK}"
            )
        if self.budget_tier not in VALID_BUDGET:
            raise ValueError(
                f"budget_tier='{self.budget_tier}' not in {VALID_BUDGET}"
            )
        if self.budget_tier == "low" and self.allow_dense_refinement:
            raise ValueError(
                "low budget routes cannot allow_dense_refinement"
            )

    @property
    def has_hard_filter(self) -> bool:
        return len(self.hard_axes) > 0

    @property
    def has_soft_rerank(self) -> bool:
        return len(self.soft_axes) > 0

    @property
    def cost_tier_value(self) -> int:
        return {"low": 1, "medium": 2, "high": 3, "full": 4}[self.budget_tier]

    def to_dict(self) -> dict:
        return {
            "route_id": self.route_id,
            "description": self.description,
            "hard_axes": list(self.hard_axes),
            "soft_axes": list(self.soft_axes),
            "candidate_topm": self.candidate_topm,
            "rerank_mode": self.rerank_mode,
            "budget_tier": self.budget_tier,
            "use_offline_index": self.use_offline_index,
            "allow_image_model_calls": self.allow_image_model_calls,
            "allow_dense_refinement": self.allow_dense_refinement,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalRoute":
        return cls(
            route_id=d["route_id"],
            description=d.get("description", ""),
            hard_axes=tuple(d.get("hard_axes", [])),
            soft_axes=tuple(d.get("soft_axes", [])),
            candidate_topm=d.get("candidate_topm", 500),
            rerank_mode=d.get("rerank_mode", "nnn_qamp"),
            budget_tier=d.get("budget_tier", "low"),
            use_offline_index=d.get("use_offline_index", True),
            allow_image_model_calls=d.get("allow_image_model_calls", False),
            allow_dense_refinement=d.get("allow_dense_refinement", False),
        )


# Semantic-only fallback (always available)
FALLBACK_ROUTE = RetrievalRoute(
    route_id="R00_semantic_only_top500",
    description="Semantic OfflineIndex only, topM=500, NNN+QAMP rerank",
    hard_axes=(),
    soft_axes=(),
    candidate_topm=500,
    rerank_mode="nnn_qamp",
    budget_tier="low",
)
