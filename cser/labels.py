"""Oracle subset labels and marginal values for CSER."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schema import DEFAULT_EXPERTS, ExpertSpec, expert_id_to_index
from .subset_executor import CSERSubsetExecutor


def enumerate_valid_subsets(
    expert_specs: Sequence[ExpertSpec] = DEFAULT_EXPERTS,
    require_mandatory: bool = True,
) -> np.ndarray:
    n = len(expert_specs)
    mandatory = np.asarray([spec.mandatory for spec in expert_specs], dtype=bool)
    masks: List[np.ndarray] = []
    for bits in itertools.product([False, True], repeat=n):
        mask = np.asarray(bits, dtype=bool)
        if require_mandatory and not np.all(mask[mandatory]):
            continue
        if not mask.any():
            continue
        masks.append(mask)
    masks.sort(key=lambda m: (int(m.sum()), tuple(m.astype(int).tolist())))
    return np.stack(masks, axis=0)


@dataclass
class CSEROracleLabels:
    expert_ids: Tuple[str, ...]
    subset_masks: np.ndarray
    qualities: np.ndarray
    marginal_values: np.ndarray
    ranks: np.ndarray
    gt_filtered: np.ndarray
    costs: np.ndarray

    @property
    def n_queries(self) -> int:
        return int(self.qualities.shape[0])

    @property
    def n_subsets(self) -> int:
        return int(self.subset_masks.shape[0])

    @property
    def n_experts(self) -> int:
        return int(len(self.expert_ids))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            expert_ids=np.asarray(self.expert_ids),
            subset_masks=self.subset_masks.astype(bool),
            qualities=self.qualities.astype(np.float32),
            marginal_values=self.marginal_values.astype(np.float32),
            ranks=self.ranks.astype(np.int32),
            gt_filtered=self.gt_filtered.astype(bool),
            costs=self.costs.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CSEROracleLabels":
        data = np.load(path, allow_pickle=True)
        return cls(
            expert_ids=tuple(str(x) for x in data["expert_ids"]),
            subset_masks=data["subset_masks"].astype(bool),
            qualities=data["qualities"].astype(np.float32),
            marginal_values=data["marginal_values"].astype(np.float32),
            ranks=data["ranks"].astype(np.int32),
            gt_filtered=data["gt_filtered"].astype(bool),
            costs=data["costs"].astype(np.float32),
        )


def _quality_from_result(metric: str, rank: int, mrr: float, recall_at: Mapping[int, int]) -> float:
    if metric == "mrr":
        return float(mrr)
    if metric in ("r1", "recall@1"):
        return float(recall_at.get(1, 0))
    if metric in ("r5", "recall@5"):
        return float(recall_at.get(5, 0))
    if metric == "rank_gain":
        return 0.0 if rank < 0 else 1.0 / float(rank + 1)
    raise ValueError(f"Unknown quality metric: {metric}")


def build_oracle_labels(
    executor: CSERSubsetExecutor,
    query_embs: np.ndarray,
    gt_video_ids: Sequence[str],
    query_contexts: Optional[Sequence[Mapping[str, object]]] = None,
    expert_specs: Sequence[ExpertSpec] = DEFAULT_EXPERTS,
    metric: str = "mrr",
    protected_masks: Optional[Sequence[np.ndarray]] = None,
) -> CSEROracleLabels:
    expert_specs = tuple(expert_specs)
    expert_ids = tuple(spec.expert_id for spec in expert_specs)
    id_to_idx = expert_id_to_index(expert_specs)
    subset_masks = enumerate_valid_subsets(expert_specs)
    subset_tuples = [
        tuple(expert_ids[i] for i, flag in enumerate(mask) if flag)
        for mask in subset_masks
    ]

    n_q = len(gt_video_ids)
    n_s = len(subset_tuples)
    n_e = len(expert_ids)
    query_contexts = query_contexts or [{} for _ in range(n_q)]
    protected_masks = protected_masks or [None for _ in range(n_q)]  # type: ignore[list-item]

    qualities = np.zeros((n_q, n_s), dtype=np.float32)
    ranks = np.full((n_q, n_s), -1, dtype=np.int32)
    gt_filtered = np.zeros((n_q, n_s), dtype=bool)
    costs = np.zeros((n_q, n_s), dtype=np.float32)

    subset_index_by_bits: Dict[Tuple[int, ...], int] = {
        tuple(mask.astype(int).tolist()): i for i, mask in enumerate(subset_masks)
    }

    for qi in range(n_q):
        for si, subset in enumerate(subset_tuples):
            result = executor.execute_subset(
                subset,
                query_embs[qi],
                gt_video_ids[qi],
                query_context=query_contexts[qi],
                protected_mask=protected_masks[qi],
            )
            ranks[qi, si] = result.rank
            gt_filtered[qi, si] = result.gt_filtered
            costs[qi, si] = result.cost
            qualities[qi, si] = _quality_from_result(
                metric, result.rank, result.mrr, result.recall_at
            )

    marginal = np.full((n_q, n_s, n_e), np.nan, dtype=np.float32)
    for si, mask in enumerate(subset_masks):
        for expert_id in expert_ids:
            ei = id_to_idx[expert_id]
            if mask[ei]:
                continue
            expanded = mask.copy()
            expanded[ei] = True
            sj = subset_index_by_bits.get(tuple(expanded.astype(int).tolist()))
            if sj is not None:
                marginal[:, si, ei] = qualities[:, sj] - qualities[:, si]

    return CSEROracleLabels(
        expert_ids=expert_ids,
        subset_masks=subset_masks,
        qualities=qualities,
        marginal_values=marginal,
        ranks=ranks,
        gt_filtered=gt_filtered,
        costs=costs,
    )
