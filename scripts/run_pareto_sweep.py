"""Pareto sweep: trace cost-accuracy curve for C-QIN vs baselines.

Varies the cascade budget threshold to produce multiple (cost, R@1) points,
then compares against fixed-cost baselines (semantic-only, rule parser, etc.)

Output: reports/aaai_final/pareto_sweep.json + pareto_sweep.csv

Usage:
    python scripts/run_pareto_sweep.py \
        --cache <msrvtt_cache.npz> \
        --csv <msrvtt_test_1k.csv> \
        --text-embs <text_embs.npy> \
        --out-dir reports/aaai_final
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from core.offline_index import OfflineIndex, VideoIndexEntry, build_protos
from core.metadata import VideoMetadata
from core.query_parser import QueryParser, QueryIntent
from core.meta_filter import MetaFilter

from routing.route_schema import FALLBACK_ROUTE
from routing.route_bank import RouteBank
from routing.route_executor import RouteExecutor, RouteResult
from routing.route_bank_builder import build_route_bank_labels
from routing.qin_model import CalibratedQIN, extract_qin_features
from routing.train_qin import train_cqin, TrainConfig
from routing.calibrate_safety import calibrate_all_axes
from routing.calibrated_planner_v2 import CalibratedPlannerV2
from routing.baselines import b0_semantic_only, b1_rule_parser, b5_always_hard_all

from metadata.noisy_metadata import inject_noise_batch, NoiseConfig


# ======================================================================
#  Cost model (normalized: MobileCLIP2 image encode = 1.0)
# ======================================================================

COST_MODEL = {
    "clip_text": 0.1,       # always called
    "clip_image": 1.0,      # reference unit
    "momentdetr": 2.0,      # CLIP ViT-B/32 + DETR head
    "insightface_det": 0.5, # SCRFD
    "arcface": 0.8,         # ArcFace-r50
    "mobilenetv3": 0.3,     # scene classifier
}


def route_cost(route) -> float:
    """Estimate cost of executing a route based on budget tier."""
    base = COST_MODEL["clip_text"]  # always pay text encode
    tier_costs = {
        "low": base + 0.0,
        "medium": base + COST_MODEL["clip_image"] * 0.5,
        "high": base + COST_MODEL["clip_image"] + COST_MODEL["momentdetr"] * 0.5,
        "full": base + COST_MODEL["clip_image"] + COST_MODEL["momentdetr"] +
                COST_MODEL["insightface_det"] + COST_MODEL["arcface"] +
                COST_MODEL["mobilenetv3"],
    }
    return tier_costs.get(route.budget_tier, 1.0)


# ======================================================================
#  Budget-constrained planner (parameterized threshold)
# ======================================================================

class BudgetConstrainedPlanner:
    """C-QIN planner that only selects routes below a cost ceiling."""

    def __init__(self, model, bank, calibration, max_cost: float,
                 soft_ratio: float = 0.6):
        self.planner = CalibratedPlannerV2(
            model, bank, calibration, soft_ratio=soft_ratio
        )
        self.max_cost = max_cost
        self.bank = bank

    def select_route(self, features, intent, executor):
        decision = self.planner.plan(features, intent)
        selected = decision.selected_route
        cost = route_cost(selected)
        if cost <= self.max_cost:
            return selected, cost
        # Fallback: pick highest-value route within budget
        # Get route values from model
        x = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            out = self.planner.model(x)
        route_values = out["route_values"][0].cpu().numpy()
        # Sort routes by predicted value (descending), pick first affordable
        order = np.argsort(-route_values)
        for idx in order:
            r = self.bank.routes[idx]
            if route_cost(r) <= self.max_cost:
                return r, route_cost(r)
        return FALLBACK_ROUTE, route_cost(FALLBACK_ROUTE)


# ======================================================================
#  Data loading (reuse from run_final_eval)
# ======================================================================

def _load_data(cache_npz, csv_path, text_embs_path, seed=42):
    rng = random.Random(seed)
    np.random.seed(seed)

    noise_cfg = NoiseConfig(time_shift_days_std=7.0, geo_missing_prob=0.3,
                             geo_wrong_region_prob=0.1)

    data = np.load(cache_npz, allow_pickle=True)
    vids = [str(x) for x in data["video_ids"]]
    pa = data["protos"].astype(np.float32)
    pa /= np.linalg.norm(pa, axis=-1, keepdims=True) + 1e-9

    from datetime import datetime, timezone
    t_min = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
    t_max = datetime(2026, 4, 20, tzinfo=timezone.utc).timestamp()
    GEO = ["coast", "mountain", "urban", "indoor_home", "rural",
            "unknown", "unknown", "unknown"]
    MOT = ["running", "walking", "stationary", "stationary", "vehicle", "unknown"]

    entries, clean = [], []
    for i, vid in enumerate(vids):
        p6 = pa[i]
        p4 = np.stack([p6[:2].mean(0), p6[2:3].mean(0),
                        p6[3:5].mean(0), p6[5:6].mean(0)])
        p4 /= np.linalg.norm(p4, axis=-1, keepdims=True) + 1e-9
        p2 = np.stack([p6[:3].mean(0), p6[3:].mean(0)])
        p2 /= np.linalg.norm(p2, axis=-1, keepdims=True) + 1e-9
        m = VideoMetadata(
            creation_time=rng.uniform(t_min, t_max),
            geo_category=rng.choice(GEO),
            motion_class=rng.choice(MOT),
            motion_confidence=rng.uniform(0.5, 1.0),
        )
        clean.append(m)
        entries.append(VideoIndexEntry(
            video_id=vid, video_path="", duration=0.0,
            frame_embs=p6, protos={2: p2, 4: p4, 6: p6}, metadata=m,
        ))
    noisy = inject_noise_batch(clean, noise_cfg, seed=seed)
    for i, nm in enumerate(noisy):
        entries[i].metadata = nm

    index = OfflineIndex(entries)
    queries, gt = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in __import__("csv").DictReader(f):
            queries.append(row["sentence"]); gt.append(row["video_id"])
    q_embs = np.load(text_embs_path).astype(np.float32)[:len(queries)]
    q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9

    # Intents
    v2m = dict(zip(vids, clean))
    parser = QueryParser()
    intents = []
    for q, g in zip(queries, gt):
        it = parser.parse(q)
        m = v2m.get(g)
        if m:
            if m.creation_time and rng.random() < 0.5:
                it.time_window = (m.creation_time - 14*86400,
                                   m.creation_time + 14*86400)
            if rng.random() < 0.5:
                if m.geo_category and m.geo_category != "unknown":
                    it.geo_categories = [m.geo_category]
                if m.motion_class and m.motion_class != "unknown":
                    it.motion_classes = [m.motion_class]
        intents.append(it)

    return index, vids, queries, gt, q_embs, intents


# ======================================================================
#  Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-embs", required=True)
    ap.add_argument("--out-dir", default="reports/aaai_final")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading data ...")
    index, vids, queries, gt, q_embs, intents = _load_data(
        args.cache, args.csv, args.text_embs, args.seed
    )
    N = len(queries)
    print(f"   N={N} queries, {index.size} videos")

    # Split: 35% train, 8% cal, rest test
    perm = np.random.permutation(N)
    n_tr = int(N * 0.35)
    n_cal = int(N * 0.08)
    tr, cal_idx, te = perm[:n_tr], perm[n_tr:n_tr+n_cal], perm[n_tr+n_cal:]
    print(f"   split: tr={len(tr)} cal={len(cal_idx)} test={len(te)}")

    # Train C-QIN
    print("[2/4] Training C-QIN ...")
    bank = RouteBank.from_yaml()
    executor = RouteExecutor(index)
    labels = build_route_bank_labels(
        index, bank, q_embs[tr], [gt[i] for i in tr],
        [intents[i] for i in tr], MetaFilter(), verbose=False,
    )
    mavail = np.array([
        sum(1 for e in index.entries if e.metadata and e.metadata.creation_time) / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.geo_category
            and e.metadata.geo_category != "unknown") / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.motion_class
            and e.metadata.motion_class != "unknown") / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.device_make) / index.size,
    ], dtype=np.float32)

    def _feats(idx_arr):
        feats = []
        for i in idx_arr:
            hits = index.search_batch(q_embs[i:i+1], top_k=20,
                                       col_beta=0.0, topm_rerank=100)[0]
            sc = np.array([s for _, s, _ in hits[:20]], dtype=np.float32)
            feats.append(extract_qin_features(
                queries[i], q_embs[i], sc, intents[i], mavail
            ))
        return np.stack(feats).astype(np.float32)

    train_feats = _feats(tr)
    model, _ = train_cqin(train_feats, labels,
                            TrainConfig(epochs=args.epochs, patience=20),
                            verbose=False)

    # Calibrate
    cal_labels = build_route_bank_labels(
        index, bank, q_embs[cal_idx], [gt[i] for i in cal_idx],
        [intents[i] for i in cal_idx], MetaFilter(), verbose=False,
    )
    cal_feats = _feats(cal_idx)
    with torch.no_grad():
        cs = model(torch.from_numpy(cal_feats).float())["safety_probs"].numpy()
    cal_res = calibrate_all_axes(cs, cal_labels.survival_labels,
                                   delta=0.10, min_accept=5)

    # Test data
    te_embs = q_embs[te]
    te_gt = [gt[i] for i in te]
    te_int = [intents[i] for i in te]

    # ── Sweep budget thresholds ──
    print("[3/4] Sweeping budget thresholds ...")
    budget_levels = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    pareto_points: List[Dict] = []

    for max_cost in budget_levels:
        bp = BudgetConstrainedPlanner(model, bank, cal_res, max_cost=max_cost)
        ranks = np.full(len(te), -1, dtype=np.int32)
        gt_filt = np.zeros(len(te), dtype=bool)
        costs = np.zeros(len(te), dtype=np.float32)

        for i in range(len(te)):
            feat = extract_qin_features(
                "", te_embs[i], np.zeros(20), te_int[i], mavail
            )
            route, cost = bp.select_route(feat, te_int[i], executor)
            res = executor.execute(route, te_embs[i], te_gt[i], te_int[i])
            ranks[i] = res.rank
            gt_filt[i] = res.gt_filtered
            costs[i] = cost

        r1 = float(((ranks >= 0) & (ranks < 1)).mean())
        r5 = float(((ranks >= 0) & (ranks < 5)).mean())
        gtf = float(gt_filt.mean())
        avg_cost = float(costs.mean())

        pareto_points.append({
            "max_budget": max_cost,
            "avg_cost": avg_cost,
            "R@1": r1,
            "R@5": r5,
            "GT_filtered": gtf,
            "method": "C-QIN",
        })
        print(f"   budget={max_cost:.1f}  avg_cost={avg_cost:.2f}  "
              f"R@1={r1*100:.1f}%  GT_f={gtf*100:.1f}%")

    # ── Fixed baselines ──
    print("[4/4] Running fixed baselines ...")
    baselines = {
        "Semantic-only": b0_semantic_only,
        "Rule parser": b1_rule_parser,
        "Always-hard-all": b5_always_hard_all,
    }
    for name, fn in baselines.items():
        ranks = np.full(len(te), -1, dtype=np.int32)
        gt_filt = np.zeros(len(te), dtype=bool)
        costs_bl = np.zeros(len(te), dtype=np.float32)
        for i in range(len(te)):
            res = fn(te_embs[i], te_gt[i], te_int[i], executor, bank)
            ranks[i] = res.rank
            gt_filt[i] = res.gt_filtered
            # Use route_cost for consistent scale with C-QIN points
            matched_route = bank.get(res.route_id) if res.route_id else FALLBACK_ROUTE
            costs_bl[i] = route_cost(matched_route) if matched_route else route_cost(FALLBACK_ROUTE)
        r1 = float(((ranks >= 0) & (ranks < 1)).mean())
        gtf = float(gt_filt.mean())
        avg_cost = float(costs_bl.mean())
        pareto_points.append({
            "max_budget": None,
            "avg_cost": avg_cost,
            "R@1": r1,
            "R@5": float(((ranks >= 0) & (ranks < 5)).mean()),
            "GT_filtered": gtf,
            "method": name,
        })
        print(f"   {name:<20} cost={avg_cost:.2f}  R@1={r1*100:.1f}%  "
              f"GT_f={gtf*100:.1f}%")

    # ── Save ──
    (out / "pareto_sweep.json").write_text(
        json.dumps(pareto_points, indent=2, default=str), encoding="utf-8"
    )
    csv_path = out / "pareto_sweep.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pareto_points[0].keys()))
        w.writeheader(); w.writerows(pareto_points)

    # ── Print Pareto table ──
    print("\n" + "=" * 70)
    print("PARETO CURVE DATA")
    print("=" * 70)
    print(f"{'Method':<20} {'Budget':>7} {'AvgCost':>8} {'R@1':>7} {'GT_f':>6}")
    print("-" * 70)
    for p in sorted(pareto_points, key=lambda x: x["avg_cost"]):
        bud = f"{p['max_budget']:.1f}" if p["max_budget"] else "fixed"
        print(f"{p['method']:<20} {bud:>7} {p['avg_cost']:>7.2f} "
              f"{p['R@1']*100:>6.1f}% {p['GT_filtered']*100:>5.1f}%")
    print("=" * 70)
    print(f"\n[saved] {out}/pareto_sweep.json")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
