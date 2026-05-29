"""Greedy budgeted CSER planner."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised in lightweight envs
    torch = None

from .conformal import difficulty_from_scores
from .schema import CSERDecision, DEFAULT_EXPERTS, ExpertSpec, expert_cost
from .subset_executor import CSERSubsetExecutor


class GreedyBudgetedSelector:
    """Greedy expert selector with conformal protection."""

    def __init__(
        self,
        model,
        expert_specs: Sequence[ExpertSpec] = DEFAULT_EXPERTS,
        conformal_gate=None,
        tau_stop: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.expert_specs = tuple(expert_specs)
        self.expert_ids = tuple(spec.expert_id for spec in self.expert_specs)
        self.conformal_gate = conformal_gate
        self.tau_stop = float(tau_stop)
        self.device = torch.device(device) if torch is not None else device
        if torch is not None and hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _mandatory(self) -> Tuple[str, ...]:
        return tuple(spec.expert_id for spec in self.expert_specs if spec.mandatory)

    def _predict_values(self, query_features: np.ndarray, selected: Sequence[str]) -> np.ndarray:
        selected_mask = np.zeros(len(self.expert_ids), dtype=np.float32)
        for expert_id in selected:
            selected_mask[self.expert_ids.index(expert_id)] = 1.0

        if self.model is None:
            return np.zeros(len(self.expert_ids), dtype=np.float32)
        if callable(self.model) and not hasattr(self.model, "forward"):
            return np.asarray(self.model(query_features, selected_mask), dtype=np.float32)
        if torch is None:
            raise RuntimeError("Torch is required for SVN model inference")

        with torch.no_grad():
            x = torch.from_numpy(np.asarray(query_features, dtype=np.float32)).unsqueeze(0).to(self.device)
            m = torch.from_numpy(selected_mask).unsqueeze(0).to(self.device)
            values = self.model(x, m)[0].detach().cpu().numpy().astype(np.float32)
        return values

    def _protected_mask(
        self,
        executor: CSERSubsetExecutor,
        query_emb: np.ndarray,
    ) -> np.ndarray:
        semantic = executor.semantic_scores(query_emb)
        if self.conformal_gate is None:
            return np.zeros(executor.store.size, dtype=bool)
        return self.conformal_gate.predict(semantic, difficulty_from_scores(semantic))

    def plan(
        self,
        query_features: np.ndarray,
        query_emb: np.ndarray,
        budget: float,
        executor: CSERSubsetExecutor,
    ) -> Tuple[CSERDecision, np.ndarray]:
        selected = list(self._mandatory())
        budget_used = sum(expert_cost(e, self.expert_specs) for e in selected)
        protected = self._protected_mask(executor, query_emb)
        step_values = []

        while True:
            remaining = float(budget) - float(budget_used)
            if remaining <= 1e-8:
                break

            values = self._predict_values(query_features, selected)
            best_id = None
            best_score = -float("inf")
            best_raw = -float("inf")
            for i, expert_id in enumerate(self.expert_ids):
                if expert_id in selected:
                    continue
                cost = expert_cost(expert_id, self.expert_specs)
                if cost > remaining + 1e-8:
                    continue
                raw_value = float(values[i])
                score = raw_value / max(cost, 1e-8)
                if score > best_score:
                    best_score = score
                    best_raw = raw_value
                    best_id = expert_id

            if best_id is None or best_raw < self.tau_stop:
                break
            selected.append(best_id)
            budget_used += expert_cost(best_id, self.expert_specs)
            step_values.append(
                {
                    "expert_id": best_id,
                    "predicted_value": float(best_raw),
                    "value_per_cost": float(best_score),
                    "budget_used": float(budget_used),
                }
            )

        decision = CSERDecision(
            selected_experts=tuple(selected),
            budget_used=float(budget_used),
            conformal_set_size=int(protected.sum()),
            step_values=step_values,
            used_fallback=len(selected) == len(self._mandatory()),
        )
        return decision, protected

    def plan_and_execute(
        self,
        query_features: np.ndarray,
        query_emb: np.ndarray,
        gt_video_id: str,
        budget: float,
        executor: CSERSubsetExecutor,
        query_context: Optional[Mapping[str, object]] = None,
    ):
        decision, protected = self.plan(query_features, query_emb, budget, executor)
        result = executor.execute_subset(
            decision.selected_experts,
            query_emb,
            gt_video_id,
            query_context=query_context,
            protected_mask=protected,
        )
        return decision, result
