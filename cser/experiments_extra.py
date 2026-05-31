"""Extended experiments E7-E10 (plan §7), real-expert version.

Built on GallerySignals + QueryExpertPriors + SVN + CSERPipeline + baselines.

* E7  scalability      — vary gallery size, measure CSER R@1 + latency + speedup
* E8  robustness       — perturb the cached expert signals, compare CSER vs cascade
* E9  expert contrib   — per-expert marginal value (SVN vs oracle), expert ranking
* E10 oracle comparison — CSER vs oracle / greedy-true / all-experts, % of oracle
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Sequence

import numpy as np

from .experts import (N_OPTIONAL, OPTIONAL_NAMES, OPTIONAL_COSTS, SEMANTIC_COST,
                      all_optional_masks, id_to_mask, mask_to_id, mask_to_names)
from .retrieval import RetrievalEngine
from .value_oracle import build_oracle_labels, OracleLabels
from .greedy import GreedyBudgetedSelector
from .baselines import FixedCascade, oracle_mask
from eval.metrics import retrieval_metrics


# ----------------------------------------------------------------------
#  E7 — scalability (gallery size sweep)
# ----------------------------------------------------------------------

def exp_e7_scalability(dataset, model, budget: float = 5.0,
                       sizes: Sequence[int] = (250, 500, 1000),
                       seed: int = 42) -> Dict:
    """Sub-sample the gallery to several sizes; measure CSER R@1 + latency.

    The GT video of each evaluated query is always kept in the sub-gallery.
    """
    rng = np.random.default_rng(seed)
    full = dataset.gallery
    N_full = full.size

    n_eval = min(dataset.n_queries, 120)
    q_ids = list(range(n_eval))
    keep = {dataset.gt_video_ids[i] for i in q_ids}
    pool = [v for v in full.video_ids if v not in keep]

    out = {}
    for size in sizes:
        size = min(size, N_full)
        n_extra = max(0, size - len(keep))
        extra = list(rng.choice(pool, size=min(n_extra, len(pool)), replace=False)) \
            if pool else []
        sub_ids = list(keep) + extra
        sub_gallery = full.subset(sub_ids)
        eng = RetrievalEngine(sub_gallery)

        priors = [dataset.query_priors[i] for i in q_ids]
        gts = [dataset.gt_video_ids[i] for i in q_ids]
        oracle = build_oracle_labels(eng, priors, gts, verbose=False)

        sel = GreedyBudgetedSelector(model, budget=budget)
        all_names = list(OPTIONAL_NAMES)
        ranks, cser_ms, full_ms = [], [], []
        for k in range(len(q_ids)):
            r = sel.select(oracle.query_feats[k])
            t0 = time.perf_counter()
            rank = eng.rank_of_gt(priors[k], gts[k], r.active_experts)
            cser_ms.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            eng.rank_of_gt(priors[k], gts[k], all_names)
            full_ms.append((time.perf_counter() - t0) * 1000.0)
            ranks.append(rank)
        rm = retrieval_metrics(np.array(ranks, np.int32))
        cl, fl = float(np.mean(cser_ms)), float(np.mean(full_ms))
        out[f"gallery={sub_gallery.size}"] = {
            "gallery_size": sub_gallery.size,
            "cser_R@1": rm["R@1"], "cser_R@5": rm["R@5"],
            "cser_latency_ms": cl, "full_pipeline_latency_ms": fl,
            "speedup": float(fl / cl) if cl > 0 else 1.0,
        }
    return out


# ----------------------------------------------------------------------
#  E8 — robustness to degraded expert signals
# ----------------------------------------------------------------------

def _perturb_gallery(gallery, level: float, seed: int):
    """Return a copy of the gallery with expert signals corrupted at `level`.

    Adds noise to highlight/face scores, randomly drops face embeddings, and
    flips a fraction of scene labels — simulating unreliable upstream models.
    """
    rng = np.random.default_rng(seed)
    g = copy.deepcopy(gallery)
    scenes = ["indoor", "outdoor", "nature", "urban", "beach", "sport"]
    for s in g.signals:
        s.highlight_score = float(np.clip(
            s.highlight_score + rng.normal(0, level), 0, 1))
        s.face_score = float(np.clip(
            s.face_score + rng.normal(0, level), 0, 1))
        if rng.random() < level:
            s.face_emb = (s.face_emb + rng.normal(0, level, s.face_emb.shape)
                          ).astype(np.float32)
        if s.scene_dist and rng.random() < level:
            s.scene_dist = {rng.choice(scenes): 1.0}
    return g


def exp_e8_robustness(dataset, model, budget: float = 5.0, seed: int = 42) -> Dict:
    levels = {"clean": 0.0, "mild": 0.1, "medium": 0.25, "heavy": 0.5}
    n_eval = min(dataset.n_queries, 150)
    q_ids = list(range(n_eval))
    priors = [dataset.query_priors[i] for i in q_ids]
    gts = [dataset.gt_video_ids[i] for i in q_ids]

    out = {}
    for lvl, amt in levels.items():
        g = dataset.gallery if amt == 0.0 else _perturb_gallery(dataset.gallery, amt, seed)
        eng = RetrievalEngine(g)
        oracle = build_oracle_labels(eng, priors, gts, verbose=False)
        sel = GreedyBudgetedSelector(model, budget=budget)
        casc = FixedCascade(budget=budget)
        cser_ranks, casc_ranks = [], []
        for k in range(len(q_ids)):
            rc = sel.select(oracle.query_feats[k])
            cser_ranks.append(eng.rank_of_gt(priors[k], gts[k], rc.active_experts))
            mk = casc.select(oracle.query_feats[k])
            casc_ranks.append(eng.rank_of_gt(priors[k], gts[k], mask_to_names(mk)))
        out[lvl] = {
            "cser_R@1": retrieval_metrics(np.array(cser_ranks, np.int32))["R@1"],
            "cascade_R@1": retrieval_metrics(np.array(casc_ranks, np.int32))["R@1"],
            "cser_GT_filtered": 0.0, "cascade_GT_filtered": 0.0,
        }
    return out


# __APPEND_E9_E10__


# ----------------------------------------------------------------------
#  E9 — expert contribution analysis
# ----------------------------------------------------------------------

def exp_e9_expert_contribution(oracle: OracleLabels, model) -> Dict:
    """Per-expert marginal value: oracle (from empty set) vs SVN prediction."""
    import torch
    V = oracle.value_matrix
    empty = V[:, 0]
    oracle_marg = np.zeros((oracle.n_queries, N_OPTIONAL))
    for e in range(N_OPTIONAL):
        oracle_marg[:, e] = V[:, 1 << e] - empty

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(oracle.query_feats.astype(np.float32))
        pred = model(x, torch.zeros(oracle.n_queries, N_OPTIONAL)).numpy()

    out = {"per_expert": {}}
    for e in range(N_OPTIONAL):
        out["per_expert"][OPTIONAL_NAMES[e]] = {
            "oracle_mean_marginal": float(oracle_marg[:, e].mean()),
            "oracle_std_marginal": float(oracle_marg[:, e].std()),
            "svn_mean_marginal": float(pred[:, e].mean()),
            "frac_queries_positive": float((oracle_marg[:, e] > 1e-4).mean()),
        }
    fo, fp = oracle_marg.ravel(), pred.ravel()
    corr = (float(np.corrcoef(fo, fp)[0, 1])
            if fo.std() > 1e-9 and fp.std() > 1e-9 else 0.0)
    out["svn_oracle_marginal_correlation"] = corr
    order = np.argsort(-oracle_marg.mean(axis=0))
    out["expert_ranking_by_value"] = [OPTIONAL_NAMES[e] for e in order]
    return out


# ----------------------------------------------------------------------
#  E10 — comparison with oracle
# ----------------------------------------------------------------------

def exp_e10_oracle_comparison(oracle: OracleLabels, model,
                              budget: float = 5.0) -> Dict:
    n = oracle.n_queries
    best = oracle.best_subset_value()
    denom = max(float(best.mean()), 1e-9)

    def realised(masks):
        return float(np.mean([oracle.value_matrix[q, mask_to_id(masks[q])]
                              for q in range(n)]))

    all_masks = [np.ones(N_OPTIONAL, dtype=bool)] * n
    oracle_masks = [oracle_mask(oracle.value_matrix[q], budget) for q in range(n)]
    greedy_true = [_greedy_true(oracle.value_matrix[q], budget) for q in range(n)]
    sel = GreedyBudgetedSelector(model, budget=budget)
    cser_masks = [sel.select(oracle.query_feats[q]).selected_mask for q in range(n)]

    methods = {
        "oracle_best_subset": (best.mean(), 1.0),
        "greedy_true_values": (realised(greedy_true), realised(greedy_true) / denom),
        "cser_svn_greedy": (realised(cser_masks), realised(cser_masks) / denom),
        "all_experts": (realised(all_masks), realised(all_masks) / denom),
        "oracle_budget_feasible": (realised(oracle_masks), realised(oracle_masks) / denom),
        "semantic_only": (float(oracle.empty_set_value().mean()),
                          float(oracle.empty_set_value().mean()) / denom),
    }
    return {k: {"mean_value": float(v), "pct_of_oracle": float(p)}
            for k, (v, p) in methods.items()}


def _greedy_true(value_row: np.ndarray, budget: float) -> np.ndarray:
    selected = np.zeros(N_OPTIONAL, dtype=bool)
    remaining = budget - SEMANTIC_COST
    while True:
        sid = mask_to_id(selected)
        best_j, best_gain = -1, 1e-9
        for j in range(N_OPTIONAL):
            if selected[j] or OPTIONAL_COSTS[j] > remaining:
                continue
            gain = value_row[sid | (1 << j)] - value_row[sid]
            if gain > best_gain:
                best_gain, best_j = gain, j
        if best_j < 0:
            break
        selected[best_j] = True
        remaining -= float(OPTIONAL_COSTS[best_j])
    return selected
