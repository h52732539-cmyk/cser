"""Route bank ablation: evaluate C-QIN with different bank compositions.

Tests: 5-semantic / 10-hard / 10-soft / 15-sem+hard / 15-sem+soft / 30-full

Usage:
    python scripts/run_route_bank_ablation.py \
        --cache <msrvtt_cache.npz> \
        --csv <msrvtt_test_1k.csv> \
        --text-embs <text_embs.npy> \
        --out-dir reports/aaai_final
"""
from __future__ import annotations

import argparse
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
from routing.calibrate_safety import calibrate_all_axes
from routing.calibrated_planner_v2 import CalibratedPlannerV2, BudgetedCascadePlanner
from core.meta_filter import MetaFilter


BANK_CONFIGS = {
    "5_semantic": "configs/route_bank_5_semantic.yaml",
    "10_hard": "configs/route_bank_10_hard.yaml",
    "10_soft": "configs/route_bank_10_soft.yaml",
    "30_full": "configs/route_bank_30.yaml",
}


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

    from scripts.run_pareto_sweep import _load_data
    index, vids, queries, gt, q_embs, intents = _load_data(
        args.cache, args.csv, args.text_embs, args.seed
    )
    N = len(queries)
    perm = np.random.permutation(N)
    n_tr = int(N * 0.35); n_cal = int(N * 0.08)
    tr, cal_idx, te = perm[:n_tr], perm[n_tr:n_tr+n_cal], perm[n_tr+n_cal:]

    mavail = np.array([
        sum(1 for e in index.entries if e.metadata and e.metadata.creation_time) / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.geo_category
            and e.metadata.geo_category != "unknown") / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.motion_class
            and e.metadata.motion_class != "unknown") / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.device_make) / index.size,
    ], dtype=np.float32)

    def _feats(idx_arr, idx_index):
        feats = []
        for i in idx_arr:
            hits = idx_index.search_batch(q_embs[i:i+1], top_k=20,
                                           col_beta=0.0, topm_rerank=100)[0]
            sc = np.array([s for _, s, _ in hits[:20]], dtype=np.float32)
            feats.append(extract_qin_features(
                queries[i], q_embs[i], sc, intents[i], mavail
            ))
        return np.stack(feats).astype(np.float32)

    results = []
    for bank_name, bank_path in BANK_CONFIGS.items():
        print(f"\n=== Bank: {bank_name} ({bank_path}) ===")
        bank = RouteBank.from_yaml(str(PROJECT_ROOT / bank_path))
        executor = RouteExecutor(index)
        print(f"   routes: {len(bank)}")

        # Train C-QIN for this bank
        labels = build_route_bank_labels(
            index, bank, q_embs[tr], [gt[i] for i in tr],
            [intents[i] for i in tr], MetaFilter(), verbose=False,
        )
        train_feats = _feats(tr, index)
        model = CalibratedQIN(input_dim=531, num_routes=len(bank))
        model, _ = train_cqin(train_feats, labels,
                                TrainConfig(epochs=args.epochs, patience=20),
                                verbose=False)

        # Calibrate
        cal_labels = build_route_bank_labels(
            index, bank, q_embs[cal_idx], [gt[i] for i in cal_idx],
            [intents[i] for i in cal_idx], MetaFilter(), verbose=False,
        )
        cal_feats = _feats(cal_idx, index)
        with torch.no_grad():
            cs = model(torch.from_numpy(cal_feats).float())["safety_probs"].numpy()
        cal_res = calibrate_all_axes(cs, cal_labels.survival_labels,
                                       delta=0.10, min_accept=5)

        # Evaluate on test
        planner = CalibratedPlannerV2(model, bank, cal_res, soft_ratio=0.6)
        cascade = BudgetedCascadePlanner(planner, bank)
        te_feats = _feats(te, index)

        ranks = np.full(len(te), -1, dtype=np.int32)
        gt_filt = np.zeros(len(te), dtype=bool)
        for i in range(len(te)):
            res = cascade.plan_and_execute(
                te_feats[i], q_embs[te[i]], gt[te[i]], intents[te[i]], executor
            )
            ranks[i] = res.rank
            gt_filt[i] = res.gt_filtered

        r1 = float(((ranks >= 0) & (ranks < 1)).mean())
        r5 = float(((ranks >= 0) & (ranks < 5)).mean())
        gtf = float(gt_filt.mean())

        row = {
            "bank": bank_name,
            "n_routes": len(bank),
            "R@1": r1,
            "R@5": r5,
            "GT_filtered": gtf,
        }
        results.append(row)
        print(f"   R@1={r1*100:.1f}%  R@5={r5*100:.1f}%  GT_f={gtf*100:.1f}%")

    # Save
    (out / "route_bank_ablation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\n[saved] {out}/route_bank_ablation.json")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Bank':<15} {'Routes':>6} {'R@1':>7} {'R@5':>7} {'GT_f':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['bank']:<15} {r['n_routes']:>6} {r['R@1']*100:>6.1f}% "
              f"{r['R@5']*100:>6.1f}% {r['GT_filtered']*100:>5.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
