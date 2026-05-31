"""Submodular Value Network (plan §4.2) + ablation variants (plan §7-E5).

Predicts the marginal value ``v(e | S, q)`` of every optional expert ``e`` given
the query ``q`` and the already-selected set ``S``. Conditioning on ``S`` is what
distinguishes this from the C-QIN MLP (which scores routes independently) and is
what lets the value function be genuinely context-dependent / submodular.

Architecture (``variant="full"``)
---------------------------------
* Query encoder:        query_feat (527-D) -> d_model
* Expert embeddings:    learnable table (K0, d_model)
* Set encoder (DeepSets): sum-pool embeddings of selected experts -> d_model
* Cross-attention:      [query ⊕ set] token attends over all K0 expert tokens
* Value head:           per-expert MLP on [expert_token ⊕ context] -> scalar

Ablation variants (E5)
----------------------
* ``"full"``                 — set-conditioned + cross-attention (default)
* ``"no_cross_attn"``        — drop attention, use pooled set repr only
* ``"no_set_conditioning"``  — ignore S entirely (independent MLP, ≈ C-QIN head)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .experts import N_OPTIONAL


VARIANTS = ("full", "no_cross_attn", "no_set_conditioning")


class SubmodularValueNetwork(nn.Module):
    def __init__(self,
                 d_query: int = 522,
                 d_model: int = 128,
                 n_experts: int = N_OPTIONAL,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 variant: str = "full") -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got '{variant}'")
        self.variant = variant
        self.n_experts = n_experts
        self.d_model = d_model

        self.query_encoder = nn.Sequential(
            nn.Linear(d_query, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, d_model),
        )
        self.expert_embeddings = nn.Embedding(n_experts, d_model)

        if variant != "no_set_conditioning":
            self.set_encoder = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        if variant == "full":
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True)
            self.attn_norm = nn.LayerNorm(d_model)

        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, query_feat: torch.Tensor,
                selected_mask: torch.Tensor) -> torch.Tensor:
        """Predict marginal value for each optional expert.

        Args:
            query_feat:    (B, d_query)
            selected_mask: (B, K0) float/bool — 1 where expert already selected
        Returns:
            (B, K0) predicted marginal values v(e | S, q).
        """
        B = query_feat.shape[0]
        q = self.query_encoder(query_feat)                       # (B, d_model)

        expert_ids = torch.arange(self.n_experts, device=query_feat.device)
        expert_emb = self.expert_embeddings(expert_ids)          # (K0, d_model)
        expert_emb_b = expert_emb.unsqueeze(0).expand(B, -1, -1)  # (B, K0, d_model)

        if self.variant == "no_set_conditioning":
            context = q                                          # ignore S
        else:
            mask = selected_mask.float().unsqueeze(-1)           # (B, K0, 1)
            sel_emb = expert_emb_b * mask                        # zero out unselected
            set_repr = self.set_encoder(sel_emb).sum(dim=1)      # (B, d_model)
            if self.variant == "full":
                # token = query+set summary attends over all expert tokens
                token = (q + set_repr).unsqueeze(1)              # (B, 1, d_model)
                attended, _ = self.cross_attn(token, expert_emb_b, expert_emb_b)
                context = self.attn_norm(attended.squeeze(1) + q + set_repr)
            else:  # no_cross_attn
                context = q + set_repr                           # (B, d_model)

        context_b = context.unsqueeze(1).expand(-1, self.n_experts, -1)
        combined = torch.cat([expert_emb_b, context_b], dim=-1)  # (B, K0, 2*d_model)
        values = self.value_head(combined).squeeze(-1)           # (B, K0)
        return values

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
