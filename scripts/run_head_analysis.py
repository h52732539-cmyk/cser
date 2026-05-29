"""C-QIN head analysis: route utilization, ECE, conflict, feature importance.

Produces:
  - Route utilization histogram (which routes are selected most)
  - Expected Calibration Error (ECE) for safety head
  - Head conflict analysis (value vs safety disagreement)
  - Permutation feature importance

Usage:
    python scripts/run_head_analysis.py \
        --cache <msrvtt_cache.npz> \
        --csv <msrvtt_test_1k.csv> \
        --text-embs <text_embs.npy> \
        --out-dir reports/aaai_final
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
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

from routing.route_bank import RouteBank
from routing.route_executor import RouteExecutor
from routing.route_bank_builder import build_route_bank_labels
from routing.qin_model import CalibratedQIN, extract_qin_features
from routing.train_qin import train_cqin, TrainConfig
from routing.calibrate_safety import calibrate_all_axes, SAFETY_AXES
from routing.calibrated_planner_v2 import CalibratedPlannerV2

from metadata.noisy_metadata import inject_noise_batch, NoiseConfig


# ======================================================================
#  ECE (Expected Calibration Error)
# ======================================================================

def compute_ece(predicted_probs: np.ndarray, true_labels: np.ndarray,
                 n_bins: int = 10) -> Dict:
    """Compute ECE and return per-bin reliability data."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    N = len(predicted_probs)
    for i in range(n_bins):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        if i == n_bins - 1:
            mask |= (predicted_probs == bins[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_data.append({"bin_center": (bins[i] + bins[i+1]) / 2,
                              "avg_confidence": 0, "avg_accuracy": 0, "count": 0})
            continue
        avg_conf = predicted_probs[mask].mean()
        avg_acc = true_labels[mask].mean()
        ece += abs(avg_conf - avg_acc) * n_in_bin / N
        bin_data.append({
            "bin_center": float((bins[i] + bins[i+1]) / 2),
            "avg_confidence": float(avg_conf),
            "avg_accuracy": float(avg_acc),
            "count": int(n_in_bin),
        })
    return {"ece": float(ece), "bins": bin_data}


# ======================================================================
#  Permutation Feature Importance
# ======================================================================

def permutation_importance(model: CalibratedQIN, features: np.ndarray,
                            labels, bank: RouteBank,
                            n_repeats: int = 5, seed: int = 42) -> Dict[str, float]:
    """Measure R@1 drop when each feature group is shuffled."""
    rng = np.random.default_rng(seed)
    model.eval()

    # Baseline R@1 (using route_value_head argmax)
    with torch.no_grad():
        out = model(torch.from_numpy(features).float())
    base_preds = out["route_values"].numpy().argmax(axis=1)
    base_r1 = float((labels.ranks[np.arange(len(base_preds)), base_preds] == 0).mean())

    # Feature groups: [CLIP 0:512, QPP 512:518, KW 518:523, META 523:527, BUDGET 527:531]
    groups = {
        "CLIP_text_emb (512D)": (0, 512),
        "QPP_statistics (6D)": (512, 518),
        "Keyword_indicators (5D)": (518, 523),
        "Meta_availability (4D)": (523, 527),
        "Budget_vector (4D)": (527, 531),
    }

    importance = {}
    for name, (start, end) in groups.items():
        drops = []
        for _ in range(n_repeats):
            feat_perm = features.copy()
            perm_idx = rng.permutation(len(features))
            feat_perm[:, start:end] = feat_perm[perm_idx, start:end]
            with torch.no_grad():
                out_p = model(torch.from_numpy(feat_perm).float())
            preds_p = out_p["route_values"].numpy().argmax(axis=1)
            r1_p = float((labels.ranks[np.arange(len(preds_p)), preds_p] == 0).mean())
            drops.append(base_r1 - r1_p)
        importance[name] = float(np.mean(drops))

    return {"base_R@1": base_r1, "importance": importance}


# ======================================================================
#  Head Conflict Analysis
# ======================================================================

def analyze_conflicts(model: CalibratedQIN, features: np.ndarray,
                       bank: RouteBank, cal_res: Dict,
                       labels) -> Dict:
    """Analyze when route_value_head and safety_head disagree."""
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(features).float())
    route_values = out["route_values"].numpy()
    safety_probs = out["safety_probs"].numpy()

    N = len(features)
    axis_to_idx = {a: i for i, a in enumerate(SAFETY_AXES)}

    n_agree_safe = 0
    n_agree_semantic = 0
    n_conflict = 0
    conflict_r1_loss = []

    for i in range(N):
        # What value head wants
        best_route_idx = int(route_values[i].argmax())
        best_route = bank.routes[best_route_idx]

        # Is it safe?
        is_safe = True
        for axis in best_route.hard_axes:
            ax_idx = axis_to_idx.get(axis)
            if ax_idx is None:
                continue
            cr = cal_res.get(axis)
            if cr is None or not cr.enabled:
                is_safe = False; break
            if safety_probs[i, ax_idx] < cr.tau:
                is_safe = False; break

        if is_safe and best_route.has_hard_filter:
            n_agree_safe += 1
        elif is_safe and not best_route.has_hard_filter:
            n_agree_semantic += 1
        else:
            n_conflict += 1
            # What's the R@1 cost of the conflict?
            # Value head's choice rank vs fallback rank
            val_rank = labels.ranks[i, best_route_idx]
            fallback_idx = bank.index_of("R00_semantic_only_top500")
            fallback_rank = labels.ranks[i, fallback_idx]
            if val_rank == 0 and fallback_rank != 0:
                conflict_r1_loss.append(1)
            else:
                conflict_r1_loss.append(0)

    return {
        "n_total": N,
        "n_agree_safe": n_agree_safe,
        "n_agree_semantic": n_agree_semantic,
        "n_conflict": n_conflict,
        "conflict_rate": n_conflict / max(N, 1),
        "conflict_r1_loss_mean": float(np.mean(conflict_r1_loss)) if conflict_r1_loss else 0.0,
        "conflict_r1_loss_total": sum(conflict_r1_loss),
    }


# ======================================================================
#  Main (reuses data loading from other scripts)
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
    rng = random.Random(args.seed); np.random.seed(args.seed)

    # ── Load + train (same as other scripts) ──
    print("[1/5] Loading + training ...")
    # (Reuse the same loading pattern)
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

    # Calibrate
    cal_labels = build_route_bank_labels(
        index, bank, q_embs[cal_idx], [gt[i] for i in cal_idx],
        [intents[i] for i in cal_idx], MetaFilter(), verbose=False,
    )
    cal_feats = _feats(cal_idx)
    with torch.no_grad():
        cal_out = model(torch.from_numpy(cal_feats).float())
        cal_safety = cal_out["safety_probs"].numpy()
    cal_res = calibrate_all_axes(cal_safety, cal_labels.survival_labels,
                                   delta=0.10, min_accept=5)

    # Test features + labels for analysis
    te_labels = build_route_bank_labels(
        index, bank, q_embs[te], [gt[i] for i in te],
        [intents[i] for i in te], MetaFilter(), verbose=False,
    )
    te_feats = _feats(te)

    # ── 2. Route utilization ──
    print("[2/5] Route utilization ...")
    planner = CalibratedPlannerV2(model, bank, cal_res, soft_ratio=0.6)
    route_counts = collections.Counter()
    for i in range(len(te)):
        decision = planner.plan(te_feats[i], intents[te[i]])
        route_counts[decision.selected_route.route_id] += 1
    route_hist = dict(route_counts.most_common())
    print(f"   top-5: {list(route_hist.items())[:5]}")

    # ── 3. ECE ──
    print("[3/5] ECE (safety head calibration quality) ...")
    with torch.no_grad():
        te_out = model(torch.from_numpy(te_feats).float())
        te_safety = te_out["safety_probs"].numpy()
    # Per-axis ECE
    ece_results = {}
    for i, axis in enumerate(SAFETY_AXES):
        survival = te_labels.survival_labels[:, i].astype(float)
        ece_results[axis] = compute_ece(te_safety[:, i], survival)
        print(f"   {axis}: ECE={ece_results[axis]['ece']:.4f}")

    # ── 4. Conflict analysis ──
    print("[4/5] Head conflict analysis ...")
    conflicts = analyze_conflicts(model, te_feats, bank, cal_res, te_labels)
    print(f"   conflict_rate={conflicts['conflict_rate']*100:.1f}%  "
          f"r1_loss_per_conflict={conflicts['conflict_r1_loss_mean']:.3f}")

    # ── 5. Feature importance ──
    print("[5/5] Permutation feature importance ...")
    fi = permutation_importance(model, te_feats, te_labels, bank)
    print(f"   base_R@1={fi['base_R@1']*100:.1f}%")
    for name, imp in sorted(fi["importance"].items(), key=lambda x: -x[1]):
        print(f"   {name:<30} ΔR@1={imp*100:+.2f}pp")

    # ── Save ──
    results = {
        "route_utilization": route_hist,
        "ece_per_axis": {k: v["ece"] for k, v in ece_results.items()},
        "ece_details": ece_results,
        "conflict_analysis": conflicts,
        "feature_importance": fi,
    }
    (out / "head_analysis.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[saved] {out}/head_analysis.json")


if __name__ == "__main__":
    main()
