"""Calibration sensitivity sweep: tau_hard × soft_ratio heatmap.

Sweeps tau_hard ∈ {0.1, 0.2, ..., 0.95} and soft_ratio ∈ {0.3, 0.4, ..., 0.8}
to produce a grid of (R@1, GT_filtered_rate) values.

Output: reports/aaai_final/calibration_heatmap.json + .csv

Usage:
    python scripts/run_calibration_sweep.py \
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
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from routing.route_bank import RouteBank
from routing.route_executor import RouteExecutor
from routing.route_bank_builder import build_route_bank_labels
from routing.qin_model import CalibratedQIN, extract_qin_features
from routing.train_qin import train_cqin, TrainConfig
from routing.calibrate_safety import CalibrationResult, SAFETY_AXES
from routing.calibrated_planner_v2 import (
    CalibratedPlannerV2, BudgetedCascadePlanner,
)
from core.meta_filter import MetaFilter


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
    np.random.seed(args.seed); random.seed(args.seed)

    # ── Load + train (reuse) ──
    print("[1/3] Loading + training C-QIN ...")
    from scripts.run_pareto_sweep import _load_data
    index, vids, queries, gt, q_embs, intents = _load_data(
        args.cache, args.csv, args.text_embs, args.seed
    )
    N = len(queries)
    perm = np.random.permutation(N)
    n_tr = int(N * 0.35); n_cal = int(N * 0.08)
    tr, cal_idx, te = perm[:n_tr], perm[n_tr:n_tr+n_cal], perm[n_tr+n_cal:]

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
                            TrainConfig(epochs=args.epochs, patience=20), verbose=False)
    print("   trained")

    # Test features
    te_feats = _feats(te)
    te_gt = [gt[i] for i in te]
    te_int = [intents[i] for i in te]
    te_embs = q_embs[te]

    # ── Sweep ──
    print("[2/3] Sweeping tau_hard × soft_ratio ...")
    tau_hard_grid = np.arange(0.1, 1.0, 0.1).tolist()
    soft_ratio_grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = []
    for tau_h in tau_hard_grid:
        for sr in soft_ratio_grid:
            # Build calibration with fixed tau_hard for all axes
            cal_res = {
                axis: CalibrationResult(axis=axis, tau=tau_h, enabled=True,
                                         n_accepted=100, n_total=100,
                                         empirical_failure_rate=0.0,
                                         ucb_failure_rate=0.0)
                for axis in SAFETY_AXES
            }
            planner = CalibratedPlannerV2(model, bank, cal_res, soft_ratio=sr)
            cascade = BudgetedCascadePlanner(planner, bank)

            ranks = np.full(len(te), -1, dtype=np.int32)
            gt_filt = np.zeros(len(te), dtype=bool)
            for i in range(len(te)):
                res = cascade.plan_and_execute(
                    te_feats[i], te_embs[i], te_gt[i], te_int[i], executor
                )
                ranks[i] = res.rank
                gt_filt[i] = res.gt_filtered

            r1 = float(((ranks >= 0) & (ranks < 1)).mean())
            gtf = float(gt_filt.mean())
            results.append({
                "tau_hard": round(tau_h, 2),
                "soft_ratio": sr,
                "R@1": r1,
                "GT_filtered": gtf,
            })

    print(f"   {len(results)} grid points evaluated")

    # ── Save ──
    print("[3/3] Saving ...")
    (out / "calibration_heatmap.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    csv_path = out / "calibration_heatmap.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tau_hard", "soft_ratio", "R@1", "GT_filtered"])
        w.writeheader(); w.writerows(results)

    # Print summary: find safe zone (GT_filtered=0 AND R@1 > 40%)
    safe_zone = [r for r in results if r["GT_filtered"] == 0 and r["R@1"] > 0.40]
    print(f"\n   Safe zone (GT_f=0% AND R@1>40%): {len(safe_zone)}/{len(results)} points")
    if safe_zone:
        best = max(safe_zone, key=lambda r: r["R@1"])
        print(f"   Best safe point: tau_hard={best['tau_hard']} "
              f"soft_ratio={best['soft_ratio']} R@1={best['R@1']*100:.1f}%")

    print(f"\n[saved] {out}/calibration_heatmap.json")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
