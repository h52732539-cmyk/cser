"""C-QIN: Calibrated Query-Intent Network.

2-head lightweight model (<100K params):
  Head 1 (route_value_head): predicts utility for each route in the bank
  Head 2 (filter_safety_head): predicts per-axis GT survival probability

Input features (~531D):
  - frozen CLIP text embedding (512D)
  - QPP statistics (6D)
  - keyword/parser indicators (5D)
  - metadata availability vector (4D)
  - budget vector (4D)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class CalibratedQIN(nn.Module):
    """2-head Calibrated Query-Intent Network."""

    def __init__(self,
                 input_dim: int = 531,
                 hidden1: int = 128,
                 hidden2: int = 64,
                 num_routes: int = 30,
                 num_safety_axes: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.route_value_head = nn.Linear(hidden2, num_routes)
        self.filter_safety_head = nn.Linear(hidden2, num_safety_axes)

        self._num_routes = num_routes
        self._num_safety = num_safety_axes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        route_values = self.route_value_head(h)
        safety_logits = self.filter_safety_head(h)
        return {
            "route_values": route_values,
            "safety_logits": safety_logits,
            "safety_probs": torch.sigmoid(safety_logits),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------
#  Feature extraction (numpy, no model calls)
# ----------------------------------------------------------------------

import re
import numpy as np


def extract_qin_features(
    query_text: str,
    query_emb: np.ndarray,
    sem_scores_top20: np.ndarray,
    intent,
    meta_availability: np.ndarray,
    budget_tier: str = "low",
) -> np.ndarray:
    """Build the ~531D feature vector for C-QIN.

    Args:
        query_text: raw query string
        query_emb: (512,) frozen CLIP text embedding
        sem_scores_top20: (20,) top-20 semantic scores from OfflineIndex
        intent: QueryIntent from QueryParser
        meta_availability: (4,) fraction of videos with [time, geo, motion, device]
        budget_tier: one of "low"/"medium"/"high"/"full"
    """
    # Group A: CLIP text embedding (512D)
    clip = np.asarray(query_emb, dtype=np.float32).ravel()[:512]
    if len(clip) < 512:
        clip = np.pad(clip, (0, 512 - len(clip)))

    # Group B: QPP statistics (6D)
    sc = np.asarray(sem_scores_top20, dtype=np.float32)
    if len(sc) < 20:
        sc = np.pad(sc, (0, 20 - len(sc)), constant_values=-1.0)
    sc = sc[:20]
    top1 = float(sc[0])
    top2 = float(sc[1]) if len(sc) > 1 else 0.0
    margin = top1 - top2
    sc_valid = sc[sc > -0.9]
    if len(sc_valid) > 0:
        ent = float(-np.sum(np.clip(sc_valid, 1e-9, 1) *
                             np.log(np.clip(sc_valid, 1e-9, 1) + 1e-12)))
        std = float(np.std(sc_valid))
        conc = top1 / (float(np.sum(sc_valid)) + 1e-9)
    else:
        ent, std, conc = 0.0, 0.0, 0.0
    qpp = np.array([top1, top2, margin, ent, std, conc], dtype=np.float32)

    # Group C: keyword indicators (5D)
    indicators = np.array([
        float(intent.time_window is not None),
        float(len(intent.geo_categories) > 0),
        float(len(intent.motion_classes) > 0),
        float(intent.device_filter is not None),
        float(bool(re.search(
            r'\d{4}|event|birthday|party|聚会|生日', query_text, re.I
        ))),
    ], dtype=np.float32)

    # Group D: metadata availability (4D)
    avail = np.asarray(meta_availability, dtype=np.float32)[:4]
    if len(avail) < 4:
        avail = np.pad(avail, (0, 4 - len(avail)))

    # Group E: budget vector (4D one-hot)
    budget_map = {"low": 0, "medium": 1, "high": 2, "full": 3}
    budget = np.zeros(4, dtype=np.float32)
    budget[budget_map.get(budget_tier, 0)] = 1.0

    return np.concatenate([clip, qpp, indicators, avail, budget])
