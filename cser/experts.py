"""Expert definitions, subset utilities, and the budgeted cost model.

The 5 frozen experts are the real models in ``tasks/real_models.py`` (with mock
fallbacks in ``tasks/mock_models.py``). The semantic encoder (e0, MobileCLIP) is
the mandatory base; the other four are optional experts whose marginal value the
Submodular Value Network learns to predict.

    e0  semantic   MobileCLIP            encode_text / encode_frames   MANDATORY
    e1  highlight   MomentDETR           score(frames) -> saliency     optional
    e2  face        SCRFD detector       detect(frames) -> (has,conf)  optional
    e3  face_id     ArcFace embedder     embed(frames) -> 512-D vec    optional
    e4  scene       MobileNetV3          classify(frames) -> label     optional

Cost model (plan §3): budget is measured in *model calls × frames*. Each expert
runs its model over the sampled frames of a candidate video, so the per-expert
cost is proportional to the number of frames it must encode. The semantic base
is the cheapest (it reads cached prototype embeddings); the heavy detectors cost
more. Costs are expressed in "expert-call units" and are configurable.

Conventions
-----------
* Experts are indexed 0..K-1. ``SEMANTIC_IDX == 0`` is always selected.
* A *selection* over the optional experts is a length-K0 boolean mask
  (K0 = K - 1). Index j refers to ``OPTIONAL_EXPERTS[j]``.
* ``selection_cost(mask)`` = semantic base + chosen optional experts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Expert:
    idx: int
    name: str           # short id used everywhere downstream
    model_key: str      # which model object drives it (clip/highlight/face/face_id/scene)
    cost: float         # compute cost (expert-call units) when selected
    mandatory: bool     # semantic base is always on


# ----------------------------------------------------------------------
#  The K=5 expert roster (plan §3: the five frozen models)
# ----------------------------------------------------------------------
#
# Costs reflect relative model-call × frame expense:
#   semantic  = 1.0  (cached prototype dot-product, cheapest)
#   scene     = 1.5  (MobileNetV3-Small, light CNN)
#   highlight = 2.0  (MomentDETR over CLIP features)
#   face      = 2.0  (SCRFD detection pass)
#   face_id   = 3.0  (SCRFD detect + ArcFace embed; depends on a face being found)

EXPERTS: Tuple[Expert, ...] = (
    Expert(0, "semantic",  "clip",      1.0, True),
    Expert(1, "highlight", "highlight", 2.0, False),
    Expert(2, "face",      "face",      2.0, False),
    Expert(3, "face_id",   "face_id",   3.0, False),
    Expert(4, "scene",     "scene",     1.5, False),
)

SEMANTIC_IDX = 0
N_EXPERTS = len(EXPERTS)
OPTIONAL_EXPERTS: Tuple[Expert, ...] = tuple(e for e in EXPERTS if not e.mandatory)
N_OPTIONAL = len(OPTIONAL_EXPERTS)                      # = 4
OPTIONAL_NAMES: Tuple[str, ...] = tuple(e.name for e in OPTIONAL_EXPERTS)
OPTIONAL_MODEL_KEYS: Tuple[str, ...] = tuple(e.model_key for e in OPTIONAL_EXPERTS)

OPTIONAL_COSTS: np.ndarray = np.array([e.cost for e in OPTIONAL_EXPERTS],
                                      dtype=np.float32)
SEMANTIC_COST: float = EXPERTS[SEMANTIC_IDX].cost


# ----------------------------------------------------------------------
#  Subset / mask utilities (operate on the K0 optional experts)
# ----------------------------------------------------------------------

def all_optional_masks() -> np.ndarray:
    """Every selection over the optional experts.

    Shape ``(2**N_OPTIONAL, N_OPTIONAL)`` boolean; row index == packed id
    (:func:`mask_to_id`), so downstream lattice navigation via ``sid | (1<<j)``
    is valid. (Do NOT build from itertools.product — its row order is reversed.)
    """
    return np.stack([id_to_mask(sid) for sid in range(1 << N_OPTIONAL)], axis=0)


def mask_to_names(mask: Sequence[bool]) -> List[str]:
    """Boolean optional-mask -> list of active expert names."""
    return [OPTIONAL_NAMES[j] for j in range(N_OPTIONAL) if mask[j]]


def mask_to_model_keys(mask: Sequence[bool]) -> List[str]:
    """Boolean optional-mask -> list of active model keys."""
    return [OPTIONAL_MODEL_KEYS[j] for j in range(N_OPTIONAL) if mask[j]]


def mask_to_id(mask: Sequence[bool]) -> int:
    """Pack a boolean optional-mask into an int in [0, 2**N_OPTIONAL)."""
    out = 0
    for j in range(N_OPTIONAL):
        if mask[j]:
            out |= (1 << j)
    return out


def id_to_mask(subset_id: int) -> np.ndarray:
    """Inverse of :func:`mask_to_id`."""
    return np.array([(subset_id >> j) & 1 for j in range(N_OPTIONAL)],
                    dtype=bool)


def selection_cost(mask: Sequence[bool]) -> float:
    """Budget consumed by a selection: semantic base + chosen optional experts."""
    m = np.asarray(mask, dtype=bool)
    return float(SEMANTIC_COST + OPTIONAL_COSTS[m].sum())


def cost_table() -> Dict[int, float]:
    """Map every subset id -> its cost (handy for budget-feasibility checks)."""
    return {mask_to_id(m): selection_cost(m) for m in all_optional_masks()}
