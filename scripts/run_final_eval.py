"""Final evaluation pipeline for AAAI submission.

Implements all 9 requirements from the final-eval specification:
  1. Freeze config snapshot
  2. Fresh 4-way split (train/cal/dev/final-test), multi-seed
  3. MeanR_survived + MeanR_global
  4. Cost/route distribution
  5. All baselines B0-B10
  6. Key ablations
  7. Metadata noise sweep
  8. Statistical tests (bootstrap + McNemar + binomial CI)
  9. Oracle gap

Usage:
    python scripts/run_final_eval.py \
        --cache <msrvtt_cache.npz> \
        --csv <msrvtt_test_1k.csv> \
        --text-embs <text_embs.npy> \
        --out-dir reports/aaai_final \
        --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from core.offline_index import OfflineIndex, VideoIndexEntry, build_protos
from core.metadata import VideoMetadata
from core.query_parser import QueryParser, QueryIntent
from core.meta_filter import MetaFilter

from routing.route_schema import RetrievalRoute, FALLBACK_ROUTE
from routing.route_bank import RouteBank
from routing.route_executor import RouteExecutor, RouteResult
from routing.route_bank_builder import (
    build_route_bank_labels, RouteBankLabels, compute_utility,
)
from routing.qin_model import CalibratedQIN, extract_qin_features
from routing.train_qin import train_cqin, TrainConfig
from routing.calibrate_safety import (
    calibrate_all_axes, save_calibration, CalibrationResult,
)
from routing.calibrated_planner import CalibratedPlanner
from routing.calibrated_planner_v2 import (
    CalibratedPlannerV2, BudgetedCascadePlanner,
)
from routing.baselines import (
    b0_semantic_only, b1_rule_parser, b2_qpp_only, B3RandomRoute,
    b4_oracle_route, b5_always_hard_all, b8_cascade,
    make_b6_uncalibrated,
)
from metadata.noisy_metadata import inject_noise_batch, NoiseConfig


# ======================================================================
#  Constants
# ======================================================================

WORST_RANK = 1001  # N+1 for MeanR_global when GT filtered


# ======================================================================
#  Data loading
# ======================================================================

def _load_index(cache_npz: str, noise_cfg: NoiseConfig, seed: int):
    rng = random.Random(seed)
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
    return OfflineIndex(entries), vids, clean


def _load_queries(csv_path: str):
    qs, gt = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in __import__("csv").DictReader(f):
            qs.append(row["sentence"]); gt.append(row["video_id"])
    return qs, gt


def _make_intents(queries, gt, clean, vids, rng):
    v2m = dict(zip(vids, clean))
    parser = QueryParser()
    out = []
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
        out.append(it)
    return out


def _meta_avail(index: OfflineIndex) -> np.ndarray:
    N = index.size
    return np.array([
        sum(1 for e in index.entries if e.metadata and e.metadata.creation_time) / N,
        sum(1 for e in index.entries if e.metadata and e.metadata.geo_category
            and e.metadata.geo_category != "unknown") / N,
        sum(1 for e in index.entries if e.metadata and e.metadata.motion_class
            and e.metadata.motion_class != "unknown") / N,
        sum(1 for e in index.entries if e.metadata and e.metadata.device_make) / N,
    ], dtype=np.float32)


def _extract_feats(indices, q_embs, queries, intents, index, mavail):
    feats = []
    for i in indices:
        hits = index.search_batch(
            q_embs[i:i+1], top_k=20, col_beta=0.0, topm_rerank=100
        )[0]
        sc = np.array([s for _, s, _ in hits[:20]], dtype=np.float32)
        feats.append(extract_qin_features(
            queries[i], q_embs[i], sc, intents[i], mavail
        ))
    return np.stack(feats).astype(np.float32)


# ======================================================================
#  Metrics (requirement 3: MeanR_survived + MeanR_global)
# ======================================================================

def _metrics(ranks: np.ndarray, gt_filtered: np.ndarray) -> Dict[str, float]:
    r = ranks.copy()
    gf = gt_filtered.astype(bool)
    N = len(r)

    # MeanR_global: filtered → WORST_RANK
    r_global = np.where(gf | (r < 0), WORST_RANK, r)
    # MeanR_survived: only non-filtered
    survived = r[(~gf) & (r >= 0)]

    return {
        "R@1": float(((r >= 0) & (r < 1)).mean()),
        "R@5": float(((r >= 0) & (r < 5)).mean()),
        "R@10": float(((r >= 0) & (r < 10)).mean()),
        "MRR": float(np.mean(1.0 / (r[(r >= 0)] + 1))) if (r >= 0).any() else 0.0,
        "MeanR_global": float(r_global.mean() + 1),
        "MeanR_survived": float(survived.mean() + 1) if len(survived) else float("inf"),
        "MedR_global": float(np.median(r_global) + 1),
        "GT_filtered_rate": float(gf.mean()),
        "n_total": int(N),
        "n_survived": int((~gf).sum()),
    }


# ======================================================================
#  Per-query record (requirement 4)
# ======================================================================

@dataclass
class QueryRecord:
    query_idx: int
    method: str
    selected_route: str = ""
    hard_filter_axes: str = ""
    soft_rerank_axes: str = ""
    cascade_triggered: bool = False
    n_stages: int = 1
    candidate_before: int = 0
    candidate_after: int = 0
    model_calls: float = 1.0
    latency_ms: float = 0.0
    gt_filtered: bool = False
    rank: int = -1


# ======================================================================
#  Method runner
# ======================================================================

def _run_method(name, fn, embs, gts, intents, executor, bank,
                 record_details: bool = False) -> Tuple[Dict, np.ndarray, List]:
    N = len(gts)
    ranks = np.full(N, -1, dtype=np.int32)
    gt_filt = np.zeros(N, dtype=bool)
    costs = np.zeros(N, dtype=np.float32)
    lats = np.zeros(N, dtype=np.float32)
    records: List[QueryRecord] = []

    for i in range(N):
        try:
            res = fn(embs[i], gts[i], intents[i], executor, bank)
            ranks[i] = res.rank
            gt_filt[i] = res.gt_filtered
            costs[i] = res.cost_proxy
            lats[i] = res.latency_ms
            if record_details:
                records.append(QueryRecord(
                    query_idx=i, method=name,
                    selected_route=res.route_id,
                    gt_filtered=res.gt_filtered,
                    rank=res.rank,
                    latency_ms=res.latency_ms,
                    model_calls=res.cost_proxy,
                    candidate_after=res.candidate_count,
                ))
        except Exception:
            ranks[i] = -1; gt_filt[i] = True

    m = _metrics(ranks, gt_filt)
    m["method"] = name
    m["avg_cost"] = float(costs.mean())
    m["avg_ms_query"] = float(lats.mean())
    return m, ranks, records


# ======================================================================
#  Statistical tests (requirement 8)
# ======================================================================

def _bootstrap(r_a, r_b, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(r_a)
    h_a = (r_a == 0).astype(float)
    h_b = (r_b == 0).astype(float)
    obs = float(h_a.mean() - h_b.mean())
    null = np.empty(n)
    for i in range(n):
        swap = rng.integers(0, 2, N).astype(bool)
        pa = np.where(swap, h_b, h_a)
        pb = np.where(swap, h_a, h_b)
        null[i] = pa.mean() - pb.mean()
    p = float(np.mean(np.abs(null) >= np.abs(obs)))
    boot = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        boot[i] = h_a[idx].mean() - h_b[idx].mean()
    return {
        "obs_diff": obs, "p_value": p,
        "ci_95": [float(np.percentile(boot, 2.5)),
                   float(np.percentile(boot, 97.5))],
        "sig_005": p < 0.05,
    }


def _mcnemar(r_a, r_b):
    ha, hb = (r_a == 0), (r_b == 0)
    a_only = int((ha & ~hb).sum())
    b_only = int((~ha & hb).sum())
    n = a_only + b_only
    if n == 0:
        return {"chi2": 0.0, "p": 1.0, "a_only": a_only, "b_only": b_only}
    chi2 = (abs(a_only - b_only) - 1) ** 2 / max(n, 1)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(1 - chi2_dist.cdf(chi2, 1))
    except ImportError:
        p = -1.0
    return {"chi2": float(chi2), "p": p, "a_only": a_only, "b_only": b_only}


def _binomial_ci_upper(k, n, alpha=0.05):
    """Clopper-Pearson upper CI for GT_filtered_rate=k/n."""
    if n == 0:
        return 1.0
    if k == 0:
        return float(1 - alpha ** (1.0 / n))
    try:
        from scipy.stats import beta
        return float(beta.ppf(1 - alpha, k + 1, n - k))
    except ImportError:
        return (k + 1.96 * np.sqrt(k * (1 - k / n))) / n


# ======================================================================
#  Noise configs (requirement 7)
# ======================================================================

NOISE_LEVELS = {
    "clean": NoiseConfig(
        time_shift_days_std=0.0, time_missing_prob=0.0,
        geo_jitter_km_std=0.0, geo_wrong_region_prob=0.0, geo_missing_prob=0.0,
        motion_flip_prob=0.0, motion_missing_prob=0.0,
        device_flip_prob=0.0, device_missing_prob=0.0,
    ),
    "mild": NoiseConfig(
        time_shift_days_std=3.0, time_missing_prob=0.05,
        geo_jitter_km_std=5.0, geo_wrong_region_prob=0.02, geo_missing_prob=0.05,
        motion_flip_prob=0.05, motion_missing_prob=0.05,
        device_flip_prob=0.02, device_missing_prob=0.05,
    ),
    "medium": NoiseConfig(
        time_shift_days_std=7.0, time_missing_prob=0.2,
        geo_jitter_km_std=20.0, geo_wrong_region_prob=0.1, geo_missing_prob=0.3,
        motion_flip_prob=0.15, motion_missing_prob=0.2,
        device_flip_prob=0.05, device_missing_prob=0.1,
    ),
    "heavy": NoiseConfig(
        time_shift_days_std=14.0, time_missing_prob=0.4,
        geo_jitter_km_std=50.0, geo_wrong_region_prob=0.2, geo_missing_prob=0.5,
        motion_flip_prob=0.3, motion_missing_prob=0.4,
        device_flip_prob=0.1, device_missing_prob=0.3,
    ),
    "missing": NoiseConfig(
        time_shift_days_std=0.0, time_missing_prob=0.8,
        geo_missing_prob=0.8, motion_missing_prob=0.8, device_missing_prob=0.8,
    ),
    "conflict": NoiseConfig(
        time_shift_days_std=30.0, time_missing_prob=0.1,
        geo_wrong_region_prob=0.5, geo_missing_prob=0.1,
        motion_flip_prob=0.5, motion_missing_prob=0.1,
    ),
}


# ======================================================================
#  Single-seed run
# ======================================================================

def run_one_seed(
    seed: int,
    cache_npz: str, csv_path: str, text_embs_path: str,
    out_dir: Path,
    noise_name: str = "medium",
    epochs: int = 200,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full pipeline for one random seed."""
    rng = random.Random(seed)
    np.random.seed(seed)

    noise_cfg = NOISE_LEVELS[noise_name]
    index, vids, clean = _load_index(cache_npz, noise_cfg, seed)
    queries, gt = _load_queries(csv_path)
    q_embs = np.load(text_embs_path).astype(np.float32)[:len(queries)]
    q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9
    intents = _make_intents(queries, gt, clean, vids, rng)
    mavail = _meta_avail(index)

    N = len(queries)
    perm = np.random.permutation(N)
    # 4-way split: train 35% / cal 8% / dev 7% / final-test 50%
    # Maximizes test power (n=500 for 1000 queries) while keeping
    # enough training data for C-QIN (350 queries × 30 routes = 10.5K pairs).
    n_tr = int(N * 0.35)
    n_cal = int(N * 0.08)
    n_dev = int(N * 0.07)
    tr = perm[:n_tr]
    cal = perm[n_tr:n_tr+n_cal]
    dev = perm[n_tr+n_cal:n_tr+n_cal+n_dev]
    te = perm[n_tr+n_cal+n_dev:]
    if verbose:
        print(f"  [seed={seed}] split: tr={len(tr)} cal={len(cal)} "
              f"dev={len(dev)} test={len(te)}")

    bank = RouteBank.from_yaml()
    executor = RouteExecutor(index)

    # ── Train ──
    labels = build_route_bank_labels(
        index, bank, q_embs[tr], [gt[i] for i in tr],
        [intents[i] for i in tr], MetaFilter(), verbose=False,
    )
    train_feats = _extract_feats(tr, q_embs, queries, intents, index, mavail)
    model, _ = train_cqin(
        train_feats, labels,
        TrainConfig(epochs=epochs, batch_size=128, patience=30),
        verbose=False,
    )

    # ── Calibrate ──
    cal_labels = build_route_bank_labels(
        index, bank, q_embs[cal], [gt[i] for i in cal],
        [intents[i] for i in cal], MetaFilter(), verbose=False,
    )
    cal_feats = _extract_feats(cal, q_embs, queries, intents, index, mavail)
    with torch.no_grad():
        cal_safety = model(torch.from_numpy(cal_feats).float())["safety_probs"].numpy()
    cal_res = calibrate_all_axes(cal_safety, cal_labels.survival_labels,
                                   delta=0.10, min_accept=5)

    # ── Build planners ──
    planner_v1 = CalibratedPlanner(model, bank, cal_res)
    planner_v2 = CalibratedPlannerV2(model, bank, cal_res, soft_ratio=0.6)
    cascade_p = BudgetedCascadePlanner(planner_v2, bank)

    def _b7(e, g, it, ex, bk):
        f = extract_qin_features("", e, np.zeros(20), it, mavail)
        return planner_v1.plan_and_execute(f, e, g, it, ex)[1]

    def _b9(e, g, it, ex, bk):
        f = extract_qin_features("", e, np.zeros(20), it, mavail)
        return planner_v2.plan_and_execute(f, e, g, it, ex)[1]

    def _b10(e, g, it, ex, bk):
        f = extract_qin_features("", e, np.zeros(20), it, mavail)
        return cascade_p.plan_and_execute(f, e, g, it, ex)

    b6_fn = make_b6_uncalibrated(model, bank)

    # ── Evaluate on FINAL-TEST only ──
    te_e = q_embs[te]
    te_g = [gt[i] for i in te]
    te_i = [intents[i] for i in te]

    methods = {
        "B0_semantic": b0_semantic_only,
        "B1_rule_parser": b1_rule_parser,
        "B5_always_hard": b5_always_hard_all,
        "B6_uncalibrated": b6_fn,
        "B7_calibrated_v1": _b7,
        "B8_cascade": b8_cascade,
        "B9_soft_fallback": _b9,
        "B10_budgeted_cascade": _b10,
        "B4_oracle": b4_oracle_route,
    }

    all_met = {}
    all_ranks = {}
    all_records = {}
    for name, fn in methods.items():
        m, ranks, recs = _run_method(
            name, fn, te_e, te_g, te_i, executor, bank,
            record_details=(name == "B10_budgeted_cascade"),
        )
        all_met[name] = m
        all_ranks[name] = ranks
        all_records[name] = recs

    # ── Ablation (requirement 6) ──
    # B10 without safety head: use uncalibrated model, wrap in cascade
    planner_nosafe = CalibratedPlannerV2(
        model, bank,
        {a: CalibrationResult(a, 0.0, True, 100, 100, 0, 0)
         for a in ("time", "geo", "motion", "device")},
        soft_ratio=0.0,
    )
    cascade_nosafe = BudgetedCascadePlanner(planner_nosafe, bank)
    def _abl_nosafe(e, g, it, ex, bk):
        f = extract_qin_features("", e, np.zeros(20), it, mavail)
        return cascade_nosafe.plan_and_execute(f, e, g, it, ex)

    # B10 without QPP features: zero out QPP dims
    def _abl_noqpp(e, g, it, ex, bk):
        f = extract_qin_features("", e, np.zeros(20), it, mavail)
        f[512:518] = 0.0  # QPP features at positions 512-517
        return cascade_p.plan_and_execute(f, e, g, it, ex)

    # B10 without keyword features: zero out keyword dims
    def _abl_nokw(e, g, it, ex, bk):
        f = extract_qin_features("", e, np.zeros(20), it, mavail)
        f[518:523] = 0.0  # keyword indicators at positions 518-522
        return cascade_p.plan_and_execute(f, e, g, it, ex)

    # B10 with random route selection
    b3_rand = B3RandomRoute(seed)
    def _abl_random(e, g, it, ex, bk):
        return b3_rand(e, g, it, ex, bk)

    ablations = {
        "ABL_no_safety": _abl_nosafe,
        "ABL_no_qpp": _abl_noqpp,
        "ABL_no_keyword": _abl_nokw,
        "ABL_random_route": _abl_random,
    }
    for name, fn in ablations.items():
        m, ranks, _ = _run_method(name, fn, te_e, te_g, te_i, executor, bank)
        all_met[name] = m
        all_ranks[name] = ranks

    # ── Significance (requirement 8) ──
    sig = {}
    pairs = [
        ("B10_budgeted_cascade", "B1_rule_parser"),
        ("B10_budgeted_cascade", "B8_cascade"),
        ("B9_soft_fallback", "B1_rule_parser"),
        ("B7_calibrated_v1", "B1_rule_parser"),
        ("B10_budgeted_cascade", "B4_oracle"),
    ]
    for a, b in pairs:
        if a in all_ranks and b in all_ranks:
            key = f"{a}_vs_{b}"
            sig[key] = {
                "bootstrap": _bootstrap(all_ranks[a], all_ranks[b], seed=seed),
                "mcnemar": _mcnemar(all_ranks[a], all_ranks[b]),
            }

    # Binomial CI for GT_filtered (requirement 8)
    for name in ("B10_budgeted_cascade", "B9_soft_fallback", "B7_calibrated_v1"):
        m = all_met.get(name)
        if m:
            k = int(m["GT_filtered_rate"] * m["n_total"])
            n = m["n_total"]
            m["GT_filt_binomial_CI_upper"] = _binomial_ci_upper(k, n)

    # ── Oracle gap (requirement 9) ──
    b0_r1 = all_met["B0_semantic"]["R@1"]
    b4_r1 = all_met["B4_oracle"]["R@1"]
    for name in ("B10_budgeted_cascade", "B9_soft_fallback", "B7_calibrated_v1"):
        m = all_met.get(name)
        if m and b4_r1 > b0_r1:
            m["oracle_gap_pct"] = (m["R@1"] - b0_r1) / (b4_r1 - b0_r1)

    return {
        "seed": seed,
        "split": {"train": len(tr), "cal": len(cal),
                   "dev": len(dev), "test": len(te)},
        "results": all_met,
        "significance": sig,
        "records": {k: [asdict(r) for r in v] for k, v in all_records.items()
                     if v},
    }


# ======================================================================
#  Noise sweep (requirement 7)
# ======================================================================

def run_noise_sweep(
    seed: int, cache_npz: str, csv_path: str, text_embs_path: str,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run B1/B10/B4 under each noise level."""
    results = {}
    for noise_name in NOISE_LEVELS:
        if verbose:
            print(f"  noise={noise_name} ...")
        rng = random.Random(seed); np.random.seed(seed)
        noise_cfg = NOISE_LEVELS[noise_name]
        index, vids, clean = _load_index(cache_npz, noise_cfg, seed)
        queries, gt = _load_queries(csv_path)
        q_embs = np.load(text_embs_path).astype(np.float32)[:len(queries)]
        q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9
        intents = _make_intents(queries, gt, clean, vids, rng)
        mavail = _meta_avail(index)
        bank = RouteBank.from_yaml()
        executor = RouteExecutor(index)

        # Quick train for this noise level
        N = len(queries)
        perm = np.random.permutation(N)
        tr = perm[:int(N*0.35)]
        cal_idx = perm[int(N*0.35):int(N*0.43)]
        te = perm[int(N*0.50):]

        labels = build_route_bank_labels(
            index, bank, q_embs[tr], [gt[i] for i in tr],
            [intents[i] for i in tr], MetaFilter(), verbose=False,
        )
        feats = _extract_feats(tr, q_embs, queries, intents, index, mavail)
        model, _ = train_cqin(feats, labels,
                                TrainConfig(epochs=100, patience=20), verbose=False)

        cal_labels = build_route_bank_labels(
            index, bank, q_embs[cal_idx], [gt[i] for i in cal_idx],
            [intents[i] for i in cal_idx], MetaFilter(), verbose=False,
        )
        cal_feats = _extract_feats(cal_idx, q_embs, queries, intents, index, mavail)
        with torch.no_grad():
            cs = model(torch.from_numpy(cal_feats).float())["safety_probs"].numpy()
        cal_res = calibrate_all_axes(cs, cal_labels.survival_labels,
                                       delta=0.10, min_accept=5)
        pv2 = CalibratedPlannerV2(model, bank, cal_res, soft_ratio=0.6)
        cp = BudgetedCascadePlanner(pv2, bank)

        def _b10(e, g, it, ex, bk):
            f = extract_qin_features("", e, np.zeros(20), it, mavail)
            return cp.plan_and_execute(f, e, g, it, ex)

        te_e, te_g, te_i = q_embs[te], [gt[i] for i in te], [intents[i] for i in te]
        row = {}
        for name, fn in [("B0", b0_semantic_only), ("B1", b1_rule_parser),
                          ("B10", _b10), ("B4", b4_oracle_route)]:
            m, _, _ = _run_method(name, fn, te_e, te_g, te_i, executor, bank)
            row[name] = m
        results[noise_name] = row
    return results


# ======================================================================
#  Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-embs", required=True)
    ap.add_argument("--out-dir", default="reports/aaai_final")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 123, 456, 789, 1024])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--skip-noise-sweep", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # ── Requirement 1: freeze config ──
    config_snapshot = {
        "seeds": args.seeds,
        "epochs": args.epochs,
        "noise_default": "medium",
        "delta": 0.10,
        "soft_ratio": 0.6,
        "split": "35/8/7/50 (train/cal/dev/test)",
        "route_bank": "configs/route_bank_30.yaml",
        "WORST_RANK": WORST_RANK,
        "cache": args.cache,
        "csv": args.csv,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    (out / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2), encoding="utf-8"
    )

    # ── Requirement 2: multi-seed runs ──
    all_seed_results = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        r = run_one_seed(
            seed, args.cache, args.csv, args.text_embs, out,
            epochs=args.epochs,
        )
        all_seed_results.append(r)

    # ── Aggregate across seeds ──
    method_names = list(all_seed_results[0]["results"].keys())
    agg = {}
    for name in method_names:
        vals = [sr["results"][name] for sr in all_seed_results
                 if name in sr["results"]]
        if not vals:
            continue
        agg_row = {"method": name}
        for key in ("R@1", "R@5", "R@10", "MRR",
                     "MeanR_global", "MeanR_survived",
                     "GT_filtered_rate", "avg_cost", "avg_ms_query"):
            vs = [v[key] for v in vals if key in v]
            if vs:
                agg_row[f"{key}_mean"] = float(np.mean(vs))
                agg_row[f"{key}_std"] = float(np.std(vs))
        for key in ("oracle_gap_pct", "GT_filt_binomial_CI_upper"):
            vs = [v.get(key) for v in vals if v.get(key) is not None]
            if vs:
                agg_row[f"{key}_mean"] = float(np.mean(vs))
        agg[name] = agg_row

    # ── Print main table ──
    print("\n" + "=" * 120)
    print(f"FINAL RESULTS — {len(args.seeds)} seeds, test n≥{all_seed_results[0]['split']['test']}")
    print("=" * 120)
    print(f"{'Method':<26} {'R@1':>8} {'R@5':>8} {'MRR':>7} "
          f"{'MnR_glb':>8} {'MnR_srv':>8} {'GT_f%':>6} {'cost':>5} "
          f"{'OrcGap':>7}")
    print("-" * 120)
    for name in method_names:
        r = agg.get(name, {})
        if not r:
            continue
        ogap = r.get("oracle_gap_pct_mean", "")
        ogap_s = f"{ogap*100:.1f}%" if isinstance(ogap, float) else ""
        print(f"{name:<26} "
              f"{r.get('R@1_mean',0)*100:>6.1f}%±{r.get('R@1_std',0)*100:.1f} "
              f"{r.get('R@5_mean',0)*100:>6.1f}% "
              f"{r.get('MRR_mean',0):>6.3f} "
              f"{r.get('MeanR_global_mean',0):>7.1f} "
              f"{r.get('MeanR_survived_mean',0):>7.1f} "
              f"{r.get('GT_filtered_rate_mean',0)*100:>5.1f}% "
              f"{r.get('avg_cost_mean',0):>5.1f} "
              f"{ogap_s:>7}")
    print("=" * 120)

    # ── Print significance (from first seed) ──
    print("\nSignificance (seed=", args.seeds[0], "):")
    sig = all_seed_results[0].get("significance", {})
    for key, v in sig.items():
        bt = v.get("bootstrap", {})
        print(f"  {key}: ΔR@1={bt.get('obs_diff',0)*100:+.1f}pp "
              f"p={bt.get('p_value',1):.4f} "
              f"CI=[{bt.get('ci_95',[0,0])[0]*100:.1f}, "
              f"{bt.get('ci_95',[0,0])[1]*100:.1f}] "
              f"sig={'YES' if bt.get('sig_005') else 'no'}")

    # ── Save everything ──
    (out / "main_results.json").write_text(
        json.dumps({"aggregated": agg,
                     "per_seed": [{k: v for k, v in sr.items() if k != "records"}
                                   for sr in all_seed_results]},
                    indent=2, default=str), encoding="utf-8")

    csv_path = out / "main_results.csv"
    rows_for_csv = list(agg.values())
    if rows_for_csv:
        all_keys = set()
        for row in rows_for_csv:
            all_keys.update(row.keys())
        all_keys = sorted(all_keys)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            for row in rows_for_csv:
                w.writerow(row)

    # ── Route distribution (B10 from first seed) ──
    recs = all_seed_results[0].get("records", {}).get("B10_budgeted_cascade", [])
    if recs:
        route_hist = collections.Counter(r["selected_route"] for r in recs)
        (out / "route_distribution.json").write_text(
            json.dumps(dict(route_hist.most_common()), indent=2), encoding="utf-8"
        )

    # ── Noise sweep (requirement 7) ──
    if not args.skip_noise_sweep:
        print("\n=== NOISE SWEEP ===")
        ns = run_noise_sweep(args.seeds[0], args.cache, args.csv, args.text_embs)
        (out / "noise_sweep.json").write_text(
            json.dumps(ns, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n{'Noise':<12} {'B0_R@1':>7} {'B1_R@1':>7} {'B10_R@1':>8} "
              f"{'B4_R@1':>7} {'B1_GTf':>7} {'B10_GTf':>8}")
        for nl, row in ns.items():
            print(f"{nl:<12} "
                  f"{row['B0']['R@1']*100:>6.1f}% "
                  f"{row['B1']['R@1']*100:>6.1f}% "
                  f"{row['B10']['R@1']*100:>7.1f}% "
                  f"{row['B4']['R@1']*100:>6.1f}% "
                  f"{row['B1']['GT_filtered_rate']*100:>6.1f}% "
                  f"{row['B10']['GT_filtered_rate']*100:>7.1f}%")

    # ── Baseline snapshot (requirement 9) ──
    b0_r1 = agg.get("B0_semantic", {}).get("R@1_mean", 0)
    b4_r1 = agg.get("B4_oracle", {}).get("R@1_mean", 0)
    b10_r1 = agg.get("B10_budgeted_cascade", {}).get("R@1_mean", 0)
    gap = (b10_r1 - b0_r1) / (b4_r1 - b0_r1) if b4_r1 > b0_r1 else 0
    (out / "baseline_snapshot.md").write_text(
        f"# Final Eval Baseline Snapshot\n\n"
        f"```json\n{json.dumps(config_snapshot, indent=2)}\n```\n\n"
        f"## Key Numbers\n"
        f"- B0 (semantic): R@1 = {b0_r1*100:.1f}%\n"
        f"- B10 (C-QIN cascade): R@1 = {b10_r1*100:.1f}%\n"
        f"- B4 (oracle): R@1 = {b4_r1*100:.1f}%\n"
        f"- Oracle gap closed: {gap*100:.1f}%\n"
        f"- Seeds: {args.seeds}\n"
        f"- Test n: {all_seed_results[0]['split']['test']}\n",
        encoding="utf-8",
    )

    dt = time.perf_counter() - t_total
    print(f"\n[DONE] total={dt/60:.1f}min")
    print(f"[saved] {out}/")


if __name__ == "__main__":
    main()
