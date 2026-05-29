"""C-QIN pipeline v2 — dual-threshold + soft fallback + delta sweep +
bootstrap significance testing.

Implements all 9 requirements:
  1. Dual-threshold calibrated routing (hard/soft/unsafe)
  2. Soft fallback routes
  3. B9: C-QIN calibrated + soft fallback
  4. B10: C-QIN calibrated + budgeted cascade
  5. Delta sweep: δ ∈ {0.00, 0.01, 0.03, 0.05, 0.10, 0.15}
  6. Utility = R@1 + MRR - cost_penalty - GT_filtered_penalty
  7. Full metrics output
  8. Paired bootstrap / McNemar test
  9. baseline_snapshot.md

Usage:
    python scripts/run_cqin_pipeline_v2.py \
        --cache <msrvtt_cache.npz> \
        --csv <msrvtt_test_1k.csv> \
        --text-embs <text_embs.npy> \
        --out-dir reports/aaai_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.offline_index import OfflineIndex, VideoIndexEntry, build_protos
from core.metadata import VideoMetadata
from core.query_parser import QueryParser, QueryIntent
from core.meta_filter import MetaFilter

from routing.route_schema import FALLBACK_ROUTE
from routing.route_bank import RouteBank
from routing.route_executor import RouteExecutor, RouteResult
from routing.route_bank_builder import build_route_bank_labels, RouteBankLabels, compute_utility
from routing.qin_model import CalibratedQIN, extract_qin_features
from routing.train_qin import train_cqin, TrainConfig
from routing.calibrate_safety import (
    calibrate_all_axes, save_calibration, load_calibration, CalibrationResult,
)
from routing.calibrated_planner import CalibratedPlanner
from routing.calibrated_planner_v2 import (
    CalibratedPlannerV2, BudgetedCascadePlanner,
)
from routing.baselines import (
    b0_semantic_only, b1_rule_parser, b2_qpp_only, B3RandomRoute,
    b4_oracle_route, b5_always_hard_all, b8_cascade,
    make_b6_uncalibrated, make_b7_calibrated,
)

from eval.metrics import retrieval_metrics, full_report
from metadata.noisy_metadata import inject_noise_batch, NoiseConfig


# ======================================================================
#  Updated utility (requirement 6)
# ======================================================================

def compute_utility_v2(rank: int, gt_filtered: bool, cost: float) -> float:
    """Utility = R@1_indicator + MRR - cost_penalty - filter_penalty."""
    if gt_filtered or rank < 0:
        return -2.0  # filter penalty
    hit1 = float(rank == 0)
    mrr_val = 1.0 / (rank + 1)
    return hit1 + mrr_val - 0.05 * cost


# ======================================================================
#  Paired bootstrap significance test (requirement 8)
# ======================================================================

def paired_bootstrap_test(ranks_a: np.ndarray, ranks_b: np.ndarray,
                            n_bootstrap: int = 10000, seed: int = 42) -> dict:
    """Paired bootstrap test on R@1 difference.

    H0: R@1(A) = R@1(B).
    Uses a shift-based permutation: under H0, swapping A/B labels per
    sample should not change the mean difference.
    """
    rng = np.random.default_rng(seed)
    N = len(ranks_a)
    r1_a = (ranks_a == 0).astype(float)
    r1_b = (ranks_b == 0).astype(float)
    obs_diff = float(r1_a.mean() - r1_b.mean())

    # Permutation test: randomly swap A/B per sample
    null_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        swap = rng.integers(0, 2, N).astype(bool)
        perm_a = np.where(swap, r1_b, r1_a)
        perm_b = np.where(swap, r1_a, r1_b)
        null_diffs[i] = perm_a.mean() - perm_b.mean()

    p_value = float(np.mean(np.abs(null_diffs) >= np.abs(obs_diff)))

    # Also compute bootstrap CI on the actual difference
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, N, N)
        boot_diffs[i] = r1_a[idx].mean() - r1_b[idx].mean()
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    return {
        "obs_diff_R@1": obs_diff,
        "p_value": p_value,
        "ci_95": [ci_low, ci_high],
        "significant_005": p_value < 0.05,
    }


def mcnemar_test(ranks_a: np.ndarray, ranks_b: np.ndarray) -> dict:
    """McNemar's test: are the disagreements between A and B symmetric?"""
    hit_a = (ranks_a == 0)
    hit_b = (ranks_b == 0)
    b_only = int(((~hit_a) & hit_b).sum())  # B correct, A wrong
    a_only = int((hit_a & (~hit_b)).sum())   # A correct, B wrong
    n = b_only + a_only
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "a_only": a_only, "b_only": b_only}
    chi2 = (abs(a_only - b_only) - 1) ** 2 / max(n, 1)
    from scipy.stats import chi2 as chi2_dist
    p = float(1 - chi2_dist.cdf(chi2, df=1))
    return {"chi2": float(chi2), "p_value": p,
            "a_only": a_only, "b_only": b_only}


# ======================================================================
#  Data loading (same as v1 pipeline)
# ======================================================================

def load_index_with_noisy_meta(cache_npz, noise_cfg, seed=42):
    rng = random.Random(seed)
    data = np.load(cache_npz, allow_pickle=True)
    vids = [str(x) for x in data["video_ids"]]
    protos_all = data["protos"].astype(np.float32)
    pa = protos_all / (np.linalg.norm(protos_all, axis=-1, keepdims=True) + 1e-9)

    from datetime import datetime, timezone
    t_min = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
    t_max = datetime(2026, 4, 20, tzinfo=timezone.utc).timestamp()
    GEO = ["coast", "mountain", "urban", "indoor_home", "rural",
            "unknown", "unknown", "unknown"]
    MOT = ["running", "walking", "stationary", "stationary", "vehicle", "unknown"]

    entries, clean_metas = [], []
    for i, vid in enumerate(vids):
        p6 = pa[i]
        p4 = np.stack([p6[:2].mean(0), p6[2:3].mean(0),
                        p6[3:5].mean(0), p6[5:6].mean(0)], axis=0)
        p4 /= np.linalg.norm(p4, axis=-1, keepdims=True) + 1e-9
        p2 = np.stack([p6[:3].mean(0), p6[3:].mean(0)], axis=0)
        p2 /= np.linalg.norm(p2, axis=-1, keepdims=True) + 1e-9
        m = VideoMetadata(
            creation_time=rng.uniform(t_min, t_max),
            geo_category=rng.choice(GEO),
            motion_class=rng.choice(MOT),
            motion_confidence=rng.uniform(0.5, 1.0),
        )
        clean_metas.append(m)
        entries.append(VideoIndexEntry(
            video_id=vid, video_path="", duration=0.0, key_ts=[],
            frame_embs=p6, protos={2: p2, 4: p4, 6: p6}, metadata=m,
        ))
    noisy = inject_noise_batch(clean_metas, noise_cfg, seed=seed)
    for i, nm in enumerate(noisy):
        entries[i].metadata = nm
    return OfflineIndex(entries=entries), vids, clean_metas


def load_queries(csv_path):
    qs, gt = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in __import__("csv").DictReader(f):
            qs.append(row["sentence"]); gt.append(row["video_id"])
    return qs, gt


def make_intents(queries, gt, clean_metas, vids, rng):
    vid_to_meta = dict(zip(vids, clean_metas))
    parser = QueryParser()
    intents = []
    for q, g in zip(queries, gt):
        it = parser.parse(q)
        m = vid_to_meta.get(g)
        if m:
            if m.creation_time and rng.random() < 0.5:
                it.time_window = (m.creation_time - 14*86400, m.creation_time + 14*86400)
            if rng.random() < 0.5:
                if m.geo_category and m.geo_category != "unknown":
                    it.geo_categories = [m.geo_category]
                if m.motion_class and m.motion_class != "unknown":
                    it.motion_classes = [m.motion_class]
        intents.append(it)
    return intents


# ======================================================================
#  Evaluation helpers
# ======================================================================

def _eval_method(name, fn, embs, gts, intents, executor, bank):
    N = len(gts)
    ranks = np.full(N, -1, dtype=np.int32)
    gt_filt = np.zeros(N, dtype=bool)
    costs = np.zeros(N, dtype=np.float32)
    lats = np.zeros(N, dtype=np.float32)
    for i in range(N):
        try:
            res = fn(embs[i], gts[i], intents[i], executor, bank)
            ranks[i] = res.rank
            gt_filt[i] = res.gt_filtered
            costs[i] = res.cost_proxy
            lats[i] = res.latency_ms
        except Exception:
            ranks[i] = -1; gt_filt[i] = True
    rm = retrieval_metrics(ranks)
    return {
        "method": name,
        **rm,
        "GT_filtered_rate": float(gt_filt.mean()),
        "avg_cost": float(costs.mean()),
        "avg_ms_query": float(lats.mean()),
        "_ranks": ranks,
    }


# ======================================================================
#  Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-embs", required=True)
    ap.add_argument("--out-dir", default="reports/aaai_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed); np.random.seed(args.seed)

    noise_cfg = NoiseConfig(time_shift_days_std=7.0, geo_missing_prob=0.3,
                             geo_wrong_region_prob=0.1)

    # ── Load ──
    print("[1/8] Loading ...")
    index, vids, clean_metas = load_index_with_noisy_meta(args.cache, noise_cfg, args.seed)
    queries, gt = load_queries(args.csv)
    q_embs = np.load(args.text_embs).astype(np.float32)[:len(queries)]
    q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9
    intents = make_intents(queries, gt, clean_metas, vids, rng)
    print(f"   N={len(queries)}")

    # ── Split ──
    N = len(queries)
    perm = np.random.permutation(N)
    n_tr, n_cal = int(N * 0.60), int(N * 0.15)
    tr_idx, cal_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr+n_cal], perm[n_tr+n_cal:]
    print(f"   split: train={len(tr_idx)} cal={len(cal_idx)} test={len(te_idx)}")

    # ── Route bank labels ──
    print("[2/8] Building route bank labels ...")
    bank = RouteBank.from_yaml()
    executor = RouteExecutor(index)
    labels = build_route_bank_labels(
        index, bank, q_embs[tr_idx], [gt[i] for i in tr_idx],
        [intents[i] for i in tr_idx], MetaFilter(),
    )
    labels.save(str(out / "route_bank_train.npz"))

    # ── Features ──
    print("[3/8] Extracting features ...")
    meta_avail = np.array([
        sum(1 for e in index.entries if e.metadata and e.metadata.creation_time) / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.geo_category and e.metadata.geo_category != "unknown") / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.motion_class and e.metadata.motion_class != "unknown") / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.device_make) / index.size,
    ], dtype=np.float32)

    def _feats(idx_arr):
        feats = []
        for i in idx_arr:
            hits = index.search_batch(q_embs[i:i+1], top_k=20, col_beta=0.0, topm_rerank=100)[0]
            scores = np.array([s for _, s, _ in hits[:20]], dtype=np.float32)
            feats.append(extract_qin_features(queries[i], q_embs[i], scores, intents[i], meta_avail))
        return np.stack(feats).astype(np.float32)

    train_feats = _feats(tr_idx)

    # ── Train ──
    print("[4/8] Training C-QIN ...")
    cfg = TrainConfig(epochs=args.epochs, batch_size=128, patience=30)
    model, _ = train_cqin(train_feats, labels, cfg, save_dir=str(out / "model"))

    # ── Calibrate (delta sweep) ──
    print("[5/8] Calibrating + delta sweep ...")
    cal_labels = build_route_bank_labels(
        index, bank, q_embs[cal_idx], [gt[i] for i in cal_idx],
        [intents[i] for i in cal_idx], MetaFilter(), verbose=False,
    )
    cal_feats = _feats(cal_idx)
    import torch
    with torch.no_grad():
        cal_out = model(torch.from_numpy(cal_feats).float())
        cal_safety = cal_out["safety_probs"].numpy()

    delta_sweep = {}
    for delta in [0.00, 0.01, 0.03, 0.05, 0.10, 0.15]:
        cal_res = calibrate_all_axes(cal_safety, cal_labels.survival_labels,
                                       delta=max(delta, 0.001), min_accept=5)
        delta_sweep[f"delta={delta:.2f}"] = {
            a: {"tau": r.tau, "enabled": r.enabled, "ucb": r.ucb_failure_rate}
            for a, r in cal_res.items()
        }

    # Use delta=0.10 as production (more permissive → better MeanR)
    cal_results = calibrate_all_axes(cal_safety, cal_labels.survival_labels,
                                       delta=0.10, min_accept=5)
    save_calibration(cal_results, str(out / "calibration.json"))
    (out / "delta_sweep.json").write_text(json.dumps(delta_sweep, indent=2), encoding="utf-8")
    print("   delta sweep saved")
    for a, r in cal_results.items():
        print(f"   {a}: tau_hard={r.tau:.3f} enabled={r.enabled}")

    # ── Build planners ──
    print("[6/8] Building planners ...")
    planner_v1 = CalibratedPlanner(model, bank, cal_results)
    planner_v2 = CalibratedPlannerV2(model, bank, cal_results, soft_ratio=0.6)
    cascade_planner = BudgetedCascadePlanner(planner_v2, bank)

    # ── B9: C-QIN + soft fallback ──
    def b9_cqin_soft(emb, gt_vid, intent, exec_, _bank):
        feat = extract_qin_features("", emb, np.zeros(20), intent, meta_avail)
        _, res = planner_v2.plan_and_execute(feat, emb, gt_vid, intent, exec_)
        return res

    # ── B10: C-QIN + budgeted cascade ──
    def b10_cqin_cascade(emb, gt_vid, intent, exec_, _bank):
        feat = extract_qin_features("", emb, np.zeros(20), intent, meta_avail)
        return cascade_planner.plan_and_execute(feat, emb, gt_vid, intent, exec_)

    # ── Evaluate all on TEST ──
    print("[7/8] Evaluating on test split ...")
    te_embs = q_embs[te_idx]
    te_gt = [gt[i] for i in te_idx]
    te_int = [intents[i] for i in te_idx]

    methods = {
        "B0_semantic_only": lambda e, g, it, ex, bk: b0_semantic_only(e, g, it, ex, bk),
        "B1_rule_parser": lambda e, g, it, ex, bk: b1_rule_parser(e, g, it, ex, bk),
        "B2_qpp_only": lambda e, g, it, ex, bk: b2_qpp_only(e, g, it, ex, bk),
        "B4_oracle": lambda e, g, it, ex, bk: b4_oracle_route(e, g, it, ex, bk),
        "B5_always_hard_all": lambda e, g, it, ex, bk: b5_always_hard_all(e, g, it, ex, bk),
        "B7_cqin_calibrated_v1": lambda e, g, it, ex, bk: (
            planner_v1.plan_and_execute(
                extract_qin_features("", e, np.zeros(20), it, meta_avail),
                e, g, it, ex)[1]
        ),
        "B8_cascade": lambda e, g, it, ex, bk: b8_cascade(e, g, it, ex, bk),
        "B9_cqin_soft_fallback": b9_cqin_soft,
        "B10_cqin_budgeted_cascade": b10_cqin_cascade,
    }

    all_results = []
    all_ranks = {}
    for name, fn in methods.items():
        print(f"  evaluating {name} ...")
        r = _eval_method(name, fn, te_embs, te_gt, te_int, executor, bank)
        all_ranks[name] = r.pop("_ranks")
        all_results.append(r)

    # ── Significance tests (requirement 8) ──
    print("[8/8] Significance tests ...")
    sig_pairs = [
        ("B9_cqin_soft_fallback", "B1_rule_parser"),
        ("B9_cqin_soft_fallback", "B8_cascade"),
        ("B10_cqin_budgeted_cascade", "B1_rule_parser"),
        ("B10_cqin_budgeted_cascade", "B8_cascade"),
        ("B7_cqin_calibrated_v1", "B1_rule_parser"),
        ("B9_cqin_soft_fallback", "B4_oracle"),
        ("B10_cqin_budgeted_cascade", "B4_oracle"),
    ]
    sig_results = {}
    for a_name, b_name in sig_pairs:
        if a_name in all_ranks and b_name in all_ranks:
            key = f"{a_name}_vs_{b_name}"
            bt = paired_bootstrap_test(all_ranks[a_name], all_ranks[b_name])
            try:
                mc = mcnemar_test(all_ranks[a_name], all_ranks[b_name])
            except Exception:
                mc = {"error": "scipy unavailable"}
            sig_results[key] = {"bootstrap": bt, "mcnemar": mc}

    # ── Print & save ──
    print("\n" + "=" * 100)
    print(f"C-QIN v2 Results (test n={len(te_idx)}, delta=0.10, soft_ratio=0.6)")
    print("=" * 100)
    print(f"{'Method':<30} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'MeanR':>7} "
          f"{'MedR':>5} {'GT_f%':>6} {'cost':>5} {'ms/q':>6}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['method']:<30} {r['R@1']*100:>5.1f}% {r['R@5']*100:>5.1f}% "
              f"{r['MRR']:>5.3f} {r['MeanR']:>7.1f} {r['MedR']:>5.1f} "
              f"{r['GT_filtered_rate']*100:>5.1f}% {r['avg_cost']:>5.1f} "
              f"{r['avg_ms_query']:>5.1f}")
    print("=" * 100)

    print("\nSignificance (paired bootstrap, p<0.05):")
    for key, v in sig_results.items():
        bt = v["bootstrap"]
        print(f"  {key}: ΔR@1={bt['obs_diff_R@1']*100:+.1f}pp  "
              f"p={bt['p_value']:.3f}  95%CI=[{bt['ci_95'][0]*100:.1f}, "
              f"{bt['ci_95'][1]*100:.1f}]  sig={'YES' if bt['significant_005'] else 'no'}")

    # Save
    (out / "main_results.json").write_text(json.dumps(all_results, indent=2, default=str))
    (out / "significance.json").write_text(json.dumps(sig_results, indent=2, default=str))

    csv_path = out / "main_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)

    # ── Baseline snapshot (requirement 9) ──
    snapshot = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "seed": args.seed,
        "split": {"train": len(tr_idx), "cal": len(cal_idx), "test": len(te_idx)},
        "noise_config": {"time_shift_std": 7.0, "geo_missing": 0.3, "geo_wrong": 0.1},
        "calibration_delta": 0.10,
        "soft_ratio": 0.6,
        "route_bank_size": len(bank),
        "epochs": args.epochs,
        "model_params": model.param_count(),
        "cache_path": args.cache,
        "csv_path": args.csv,
    }
    (out / "baseline_snapshot.md").write_text(
        f"# Baseline Snapshot\n\n```json\n{json.dumps(snapshot, indent=2)}\n```\n"
        f"\n## Results\n\n"
        + "\n".join(f"- {r['method']}: R@1={r['R@1']*100:.1f}% MeanR={r['MeanR']:.1f} GT_filt={r['GT_filtered_rate']*100:.1f}%"
                     for r in all_results)
        + f"\n\n## Delta Sweep\n\n```json\n{json.dumps(delta_sweep, indent=2)}\n```\n",
        encoding="utf-8",
    )

    print(f"\n[saved] {out}/main_results.csv")
    print(f"[saved] {out}/significance.json")
    print(f"[saved] {out}/baseline_snapshot.md")
    print(f"[saved] {out}/delta_sweep.json")


if __name__ == "__main__":
    main()
