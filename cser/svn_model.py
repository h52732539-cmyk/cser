"""Submodular Value Network for CSER."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubmodularValueNetwork(nn.Module):
    """Predict marginal value for each expert given query and selected set."""

    def __init__(
        self,
        query_dim: int,
        n_experts: int = 5,
        d_expert: int = 64,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.n_experts = int(n_experts)
        self.expert_embeddings = nn.Embedding(n_experts, d_expert)

        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.expert_encoder = nn.Sequential(
            nn.Linear(d_expert, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.context_norm = nn.LayerNorm(hidden)
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 3 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, query_features: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        if query_features.ndim != 2:
            raise ValueError("query_features must have shape (B, D)")
        if selected_mask.ndim != 2:
            raise ValueError("selected_mask must have shape (B, K)")
        if selected_mask.shape[1] != self.n_experts:
            raise ValueError("selected_mask width must equal n_experts")

        selected_mask = selected_mask.float()
        batch = query_features.shape[0]
        q = self.query_encoder(query_features.float())

        expert_emb = self.expert_embeddings.weight
        expert_h = self.expert_encoder(expert_emb)
        selected_repr = selected_mask @ expert_h
        selected_count = selected_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        selected_repr = selected_repr / selected_count
        context = self.context_norm(q + selected_repr)

        context_exp = context.unsqueeze(1).expand(batch, self.n_experts, -1)
        q_exp = q.unsqueeze(1).expand(batch, self.n_experts, -1)
        expert_exp = expert_h.unsqueeze(0).expand(batch, -1, -1)
        selected_bit = selected_mask.unsqueeze(-1)
        features = torch.cat([context_exp, q_exp, expert_exp, selected_bit], dim=-1)
        values = self.value_head(features).squeeze(-1)
        return values

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


def submodularity_penalty(
    pred_smaller_set: torch.Tensor,
    pred_larger_set: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalty for violations v(e|larger S) <= v(e|smaller S)."""
    penalty = F.relu(pred_larger_set - pred_smaller_set)
    if candidate_mask is not None:
        mask = candidate_mask.bool()
        if not bool(mask.any()):
            return penalty.sum() * 0.0
        penalty = penalty[mask]
    return penalty.mean() if penalty.numel() else pred_larger_set.sum() * 0.0


def non_negative_penalty(values: torch.Tensor, candidate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    penalty = F.relu(-values)
    if candidate_mask is not None:
        mask = candidate_mask.bool()
        if not bool(mask.any()):
            return penalty.sum() * 0.0
        penalty = penalty[mask]
    return penalty.mean() if penalty.numel() else values.sum() * 0.0
