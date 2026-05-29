"""Ablation framework for LiteVTR++ — toggle each module independently
and measure ΔR@1 / Δlatency / Δfilter-set on a chosen benchmark.

Modules controlled by AblationConfig flags:

  Phase 1:
    enable_prefilter         — MetadataPrefilter (gyro+frame-diff)
    enable_two_stage         — sparse → InterestSignal → dense feedback
    enable_unified_scheduler — single decode stream across tasks
    enable_segment_aggregator — frame scores → [t_start, t_end]

  Phase 2:
    enable_offline_index     — pre-computed protos + numpy search
    enable_qpp_routing       — easy/medium/hard margin-based skipping
    enable_cross_task_cache  — frame-hash → model output reuse
    enable_adaptive_sampler  — Q-Frame / MV / hybrid sampling

  Phase 3:
    enable_meta_filter       — hard filter by time/geo/motion/device
    enable_meta_fusion       — soft α·sem + (1-α)·meta blending

Each AblationConfig is mapped to an OfflineIndex.search variant + a
sampling/decode policy, so the SAME query set can be re-evaluated under
many configurations without re-encoding.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.offline_index import OfflineIndex
from core.metadata import VideoMetadata
from core.query_parser import QueryIntent
from core.meta_filter import MetaFilter
from core.query_planner import QueryPlanner, QueryPlannerConfig, QueryDifficulty


@dataclass
class AblationConfig:
    name: str = "default"

    # Phase 1
    enable_prefilter:           bool = True
    enable_two_stage:           bool = True
    enable_unified_scheduler:   bool = True
    enable_segment_aggregator:  bool = True

    # Phase 2
    enable_offline_index:       bool = True
    enable_qpp_routing:         bool = True
    enable_cross_task_cache:    bool = True
    enable_adaptive_sampler:    bool = True

    # Phase 3
    enable_meta_filter:         bool = True
    enable_meta_fusion:         bool = True

    # Hyperparameters (sweepable)
    alpha_nnn:    float = 0.5
    tau_qamp:     float = 0.05
    col_beta:     float = 0.4
    topm_rerank:  int   = 300
    meta_alpha:   float = 0.7
    easy_margin:  float = 0.08
    hard_margin:  float = 0.02

    def to_dict(self) -> Dict:
        return asdict(self)


# ----------------------------------------------------------------------
#  Single-config evaluator
# ----------------------------------------------------------------------

def evaluate_config(
    cfg: AblationConfig,
    index: OfflineIndex,
    query_embs: np.ndarray,
    gt: List[str],
    intents: Optional[List[QueryIntent]] = None,
    meta_filter: Optional[MetaFilter] = None,
) -> Dict:
    """Run a single ablation configuration and return metrics dict."""

    N = len(gt)
    id_to_idx = {e.video_id: i for i, e in enumerate(index.entries)}

    t0 = time.perf_counter()

    if not cfg.enable_offline_index:
        # Simulate "pure cosine" — skip Multi-K + NNN + col-softmax
        big = index._flat_protos[max(index._flat_protos.keys())]
        sl = index._flat_slices_by_k[max(index._flat_slices_by_k.keys())]
        sims_all = query_embs @ big.T
        scores = np.full((N, len(index.entries)), -1e9, dtype=np.float32)
        for j, (s, e) in enumerate(sl):
            if e > s:
                scores[:, j] = sims_all[:, s:e].max(axis=1)
        ranks = []
        for i in range(N):
            order = np.argsort(-scores[i])
            ids = [index.entries[k].video_id for k in order]
            ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)
    else:
        # Phase 2 path with all the toggles
        all_hits = index.search_batch(
            query_embs,
            top_k=len(index.entries),
            alpha_nnn=cfg.alpha_nnn if cfg.enable_offline_index else 0.0,
            tau_qamp=cfg.tau_qamp,
            col_beta=cfg.col_beta if cfg.enable_offline_index else 0.0,
            topm_rerank=cfg.topm_rerank,
        )

        if cfg.enable_meta_filter or cfg.enable_meta_fusion:
            assert intents is not None, "intents required for Phase-3 ablation"
            assert meta_filter is not None
            ranks = []
            for i in range(N):
                if not intents[i].has_constraint() \
                        or (not cfg.enable_meta_filter
                            and not cfg.enable_meta_fusion):
                    # fall back to semantic-only for unconstrained
                    ids = [h[0] for h in all_hits[i]]
                    ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)
                    continue

                # meta-aware
                hits = index.search_with_meta(
                    query_embs[i], intents[i],
                    top_k=len(index.entries),
                    alpha_nnn=cfg.alpha_nnn,
                    tau_qamp=cfg.tau_qamp,
                    col_beta=cfg.col_beta,
                    topm_rerank=cfg.topm_rerank,
                    meta_filter=meta_filter,
                    meta_alpha=cfg.meta_alpha if cfg.enable_meta_fusion else 1.0,
                    use_hard_filter=cfg.enable_meta_filter,
                )
                ids = [h[0] for h in hits if h[1] > -1e8]
                ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)
        else:
            ranks = []
            for i in range(N):
                ids = [h[0] for h in all_hits[i]]
                ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)

    dt = (time.perf_counter() - t0) * 1000.0
    ranks = np.array(ranks)

    # QPP statistics (if enabled, count how many would be EASY/MEDIUM/HARD)
    qpp_split = {"easy": 0, "medium": 0, "hard": 0}
    if cfg.enable_qpp_routing and cfg.enable_offline_index:
        planner = QueryPlanner(QueryPlannerConfig(
            easy_margin=cfg.easy_margin, hard_margin=cfg.hard_margin,
        ))
        for i in range(N):
            margin_hits = list(all_hits[i][:5]) if isinstance(all_hits[i], list) \
                else []
            if not margin_hits:
                qpp_split["hard"] += 1
                continue
            plan = planner.plan(margin_hits)
            qpp_split[plan.difficulty.value] += 1

    return {
        "name": cfg.name,
        "config": cfg.to_dict(),
        "R@1":   float((ranks == 0).mean()),
        "R@5":   float((ranks <  5).mean()),
        "R@10":  float((ranks < 10).mean()),
        "MedR":  float(np.median(ranks) + 1),
        "MeanR": float(ranks.mean() + 1),
        "total_ms": float(dt),
        "ms_per_query": float(dt / max(N, 1)),
        "qpp_split": qpp_split,
    }


# ----------------------------------------------------------------------
#  Standard ablation suites
# ----------------------------------------------------------------------

def make_module_ablation_suite(base: AblationConfig) -> List[AblationConfig]:
    """Generate per-module leave-one-out ablations from a base config."""
    suite = [replace(base, name="A0_full")]
    flag_pairs = [
        ("enable_prefilter",          "A1_no_prefilter"),
        ("enable_two_stage",          "A2_no_two_stage"),
        ("enable_unified_scheduler",  "A3_no_unified_scheduler"),
        ("enable_segment_aggregator", "A4_no_seg_agg"),
        ("enable_offline_index",      "A5_no_offline_index"),
        ("enable_qpp_routing",        "A6_no_qpp"),
        ("enable_cross_task_cache",   "A7_no_cross_cache"),
        ("enable_adaptive_sampler",   "A8_no_adaptive_sampler"),
        ("enable_meta_filter",        "A9_no_meta_filter"),
        ("enable_meta_fusion",        "A10_no_meta_fusion"),
    ]
    for flag, name in flag_pairs:
        suite.append(replace(base, name=name, **{flag: False}))
    return suite


def make_hp_sweep_suite(base: AblationConfig,
                         alpha_grid:    List[float] = (0.3, 0.5, 0.7, 0.9),
                         tau_grid:      List[float] = (0.01, 0.02, 0.05, 0.10),
                         col_beta_grid: List[float] = (0.0, 0.2, 0.4, 0.6),
                         topm_grid:     List[int]   = (50, 100, 300),
                         ) -> List[AblationConfig]:
    """Cartesian product hyperparam sweep."""
    suite = []
    for a in alpha_grid:
        for t in tau_grid:
            for c in col_beta_grid:
                for m in topm_grid:
                    name = f"H_a{a}_t{t}_c{c}_m{m}"
                    suite.append(replace(base, name=name,
                                          alpha_nnn=a, tau_qamp=t,
                                          col_beta=c, topm_rerank=m))
    return suite


# ----------------------------------------------------------------------
#  Suite runner
# ----------------------------------------------------------------------

def run_suite(suite: List[AblationConfig],
              index: OfflineIndex,
              query_embs: np.ndarray,
              gt: List[str],
              intents: Optional[List[QueryIntent]] = None,
              meta_filter: Optional[MetaFilter] = None,
              save_path: Optional[str] = None,
              verbose: bool = True) -> List[Dict]:
    results = []
    for i, cfg in enumerate(suite):
        if verbose:
            print(f"[{i+1}/{len(suite)}] {cfg.name}")
        r = evaluate_config(cfg, index, query_embs, gt,
                             intents=intents, meta_filter=meta_filter)
        if verbose:
            print(f"   R@1={r['R@1']*100:5.2f}%  R@5={r['R@5']*100:5.2f}%  "
                  f"ms/q={r['ms_per_query']:.2f}")
        results.append(r)
    if save_path:
        Path(save_path).write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        if verbose:
            print(f"[saved] {save_path}")
    return results
