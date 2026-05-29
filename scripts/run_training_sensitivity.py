"""Training data size sensitivity: how many queries does C-QIN need?

Trains C-QIN with {50, 100, 200, 350, 500} training queries and measures
R@1 on a fixed test set. Demonstrates learning curve saturation.

Usage:
    python scripts/run_training_sensitivity.py \
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


TRAIN_SIZES = [50, 100, 200, 350, 500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-embs", required=True)
    ap.add_argument("--out-dir", default="reports/aaai_final")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    from scripts.run_pareto_sweep import _load_data
    index, vids, queries, gt, q_embs, intents = _load_data(
        args.cache, args.csv, args.text_embs, args.seed
    )
    N = len(queries)
    bank = RouteBank.from_yaml()

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

    results = []
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    for n_train in TRAIN_SIZES:
        seed_r1s = []
        for seed in seeds:
            np.random.seed(seed); random.seed(seed)
            perm = np.random.permutation(N)
            tr = perm[:n_train]
            cal_idx = perm[n_train:n_train + 80]
            te = perm[n_train + 80:]

            executor = RouteExecutor(index)
            labels = build_route_bank_labels(
                index, bank, q_embs[tr], [gt[i] for i in tr],
                [intents[i] for i in tr], MetaFilter(), verbose=False,
            )
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

            # Evaluate
            planner = CalibratedPlannerV2(model, bank, cal_res, soft_ratio=0.6)
            cascade = BudgetedCascadePlanner(planner, bank)
            te_feats = _feats(te)

            ranks = np.full(len(te), -1, dtype=np.int32)
            for i in range(len(te)):
                res = cascade.plan_and_execute(
                    te_feats[i], q_embs[te[i]], gt[te[i]], intents[te[i]], executor
                )
                ranks[i] = res.rank
            r1 = float(((ranks >= 0) & (ranks < 1)).mean())
            seed_r1s.append(r1)

        row = {
            "n_train": n_train,
            "R@1_mean": float(np.mean(seed_r1s)),
            "R@1_std": float(np.std(seed_r1s)),
            "R@1_per_seed": seed_r1s,
        }
        results.append(row)
        print(f"  n_train={n_train:4d}  R@1={row['R@1_mean']*100:.1f}%"
              f"±{row['R@1_std']*100:.1f}")

    # Save
    (out / "training_sensitivity.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\n[saved] {out}/training_sensitivity.json")

    # Summary
    print("\n" + "=" * 50)
    print(f"{'n_train':>8} {'R@1':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['n_train']:>8} {r['R@1_mean']*100:>8.1f}%±{r['R@1_std']*100:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
