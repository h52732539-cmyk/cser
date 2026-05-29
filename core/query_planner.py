"""Query planner: decide the minimum amount of work per query.

Using your QPP v4 finding: the single best single-feature threshold on
`margin` captures most of the achievable selective-reranking gain. We
reuse that as the online router here — no learned model needed.

Decision flow:

    1. Run offline-index search → (top_k, margin).
    2. If margin > margin_easy      → return (no Stage-2)
    3. Elif margin > margin_hard    → light Stage-2 on top-3 videos
    4. Else (genuinely ambiguous)   → full Stage-2 + MomentDETR refine

This keeps the expensive model calls proportional to query difficulty
without touching Huawei model weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class QueryDifficulty(Enum):
    EASY = "easy"       # margin > easy_thr   → index-only result
    MEDIUM = "medium"   # margin > hard_thr   → re-score top candidates
    HARD = "hard"       # otherwise           → full 2-stage refinement


@dataclass
class QueryPlan:
    difficulty: QueryDifficulty
    top_candidates: List[str]          # video_ids to refine
    refine_top_n: int                  # how many to re-encode
    run_momentdetr: bool
    margin: float


@dataclass
class QueryPlannerConfig:
    easy_margin: float = 0.08          # calibrated empirically
    hard_margin: float = 0.02
    medium_refine_top_n: int = 3
    hard_refine_top_n: int = 10
    easy_return_top_k: int = 5


class QueryPlanner:
    def __init__(self, config: Optional[QueryPlannerConfig] = None) -> None:
        self.cfg = config or QueryPlannerConfig()

    # ------------------------------------------------------------------

    def plan(self, search_results: List[Tuple[str, float, float]]) -> QueryPlan:
        """`search_results` := [(video_id, score, margin)], sorted desc."""
        if not search_results:
            return QueryPlan(
                difficulty=QueryDifficulty.HARD,
                top_candidates=[], refine_top_n=0,
                run_momentdetr=False, margin=0.0,
            )

        _, _, margin = search_results[0]

        if margin >= self.cfg.easy_margin:
            ids = [v for v, _, _ in search_results[:self.cfg.easy_return_top_k]]
            return QueryPlan(
                difficulty=QueryDifficulty.EASY,
                top_candidates=ids,
                refine_top_n=0,
                run_momentdetr=False,
                margin=margin,
            )

        if margin >= self.cfg.hard_margin:
            n = self.cfg.medium_refine_top_n
            ids = [v for v, _, _ in search_results[:n]]
            return QueryPlan(
                difficulty=QueryDifficulty.MEDIUM,
                top_candidates=ids,
                refine_top_n=n,
                run_momentdetr=False,
                margin=margin,
            )

        n = self.cfg.hard_refine_top_n
        ids = [v for v, _, _ in search_results[:n]]
        return QueryPlan(
            difficulty=QueryDifficulty.HARD,
            top_candidates=ids,
            refine_top_n=n,
            run_momentdetr=True,
            margin=margin,
        )

    # ------------------------------------------------------------------

    def summarize(self, plans: List[QueryPlan]) -> dict:
        from collections import Counter
        c = Counter(p.difficulty.value for p in plans)
        n = max(len(plans), 1)
        return {
            "n_queries": len(plans),
            "easy_pct":   100.0 * c["easy"]   / n,
            "medium_pct": 100.0 * c["medium"] / n,
            "hard_pct":   100.0 * c["hard"]   / n,
            "avg_margin": float(sum(p.margin for p in plans) / n) if plans else 0.0,
        }
