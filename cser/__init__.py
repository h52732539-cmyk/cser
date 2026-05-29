"""CSER: Conformal Submodular Expert Routing.

This package is intentionally self-contained so the experimental CSER code
does not change the existing C-QIN implementation.
"""

from .schema import (
    CSERDecision,
    CSERSubsetResult,
    DEFAULT_EXPERTS,
    DEFAULT_EXPERT_IDS,
    ExpertScore,
    ExpertSpec,
)

__all__ = [
    "CSERDecision",
    "CSERSubsetResult",
    "DEFAULT_EXPERTS",
    "DEFAULT_EXPERT_IDS",
    "ExpertScore",
    "ExpertSpec",
]
