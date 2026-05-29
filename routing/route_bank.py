"""Route bank — load, validate, and manage the fixed set of candidate routes."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .route_schema import RetrievalRoute, FALLBACK_ROUTE


DEFAULT_BANK_PATH = Path(__file__).resolve().parent.parent / "configs" / "route_bank_30.yaml"


class RouteBank:
    """A fixed vocabulary of ~30 retrieval routes."""

    def __init__(self, routes: List[RetrievalRoute]) -> None:
        self._routes = list(routes)
        self._by_id: Dict[str, RetrievalRoute] = {}
        for r in self._routes:
            if r.route_id in self._by_id:
                raise ValueError(f"Duplicate route_id: {r.route_id}")
            self._by_id[r.route_id] = r
        if FALLBACK_ROUTE.route_id not in self._by_id:
            self._routes.insert(0, FALLBACK_ROUTE)
            self._by_id[FALLBACK_ROUTE.route_id] = FALLBACK_ROUTE

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "RouteBank":
        p = Path(path) if path else DEFAULT_BANK_PATH
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        routes = [RetrievalRoute.from_dict(d) for d in data["routes"]]
        return cls(routes)

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self):
        return iter(self._routes)

    def __getitem__(self, route_id: str) -> RetrievalRoute:
        return self._by_id[route_id]

    def get(self, route_id: str) -> Optional[RetrievalRoute]:
        return self._by_id.get(route_id)

    @property
    def ids(self) -> List[str]:
        return [r.route_id for r in self._routes]

    @property
    def routes(self) -> List[RetrievalRoute]:
        return list(self._routes)

    @property
    def fallback(self) -> RetrievalRoute:
        return self._by_id[FALLBACK_ROUTE.route_id]

    def routes_with_hard_axis(self, axis: str) -> List[RetrievalRoute]:
        return [r for r in self._routes if axis in r.hard_axes]

    def routes_by_budget(self, tier: str) -> List[RetrievalRoute]:
        return [r for r in self._routes if r.budget_tier == tier]

    def index_of(self, route_id: str) -> int:
        for i, r in enumerate(self._routes):
            if r.route_id == route_id:
                return i
        raise KeyError(f"route_id '{route_id}' not in bank")

    def summary(self) -> Dict:
        from collections import Counter
        return {
            "n_routes": len(self),
            "budget_dist": dict(Counter(r.budget_tier for r in self._routes)),
            "n_with_hard_filter": sum(1 for r in self._routes if r.has_hard_filter),
            "n_with_soft_rerank": sum(1 for r in self._routes if r.has_soft_rerank),
            "n_dense_refine": sum(1 for r in self._routes if r.allow_dense_refinement),
        }
