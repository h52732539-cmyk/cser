"""Set-value predictor for budgeted expert subset selection.

The legacy SVN predicts per-expert marginals and then relies on greedy search.
This model predicts the final retrieval value ``F(q, S)`` for every optional
expert subset directly. With four optional experts the full lattice has only
16 subsets, so inference can enumerate all feasible masks without approximation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .experts import N_OPTIONAL


class SetValueNetwork(nn.Module):
    """Predict one value per optional-expert subset."""

    def __init__(self,
                 d_query: int = 522,
                 d_model: int = 128,
                 n_experts: int = N_OPTIONAL,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        self.n_subsets = 1 << self.n_experts
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_query, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, self.n_subsets),
        )

    def forward(self, query_feat: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 2**K0)`` predicted subset values."""
        return self.net(query_feat)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
