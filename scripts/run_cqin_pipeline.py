"""End-to-end C-QIN pipeline: build route bank → train → calibrate → eval.

Usage:
    python scripts/run_cqin_pipeline.py \
        --cache <msrvtt_cache.npz> \
        --csv   <msrvtt_test_1k.csv> \
        --text-embs <text_embs.npy> \
        --out-dir reports/aaai_main
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
from routing.route_executor import RouteExecutor
from routing.route_bank_builder import build_route_bank_labels, RouteBankLabels
from routing.qin_model import CalibratedQIN, extract_qin_features
from routing.train_qin import train_cqin, TrainConfig
from routing.calibrate_safety import calibrate_all_axes, save_calibration
from routing.calibrated_planner import CalibratedPlanner
from routing.baselines import get_all_baselines

from eval.eval_planner import evaluate_all_methods
from eval.metrics import retrieval_metrics

from metadata.noisy_metadata import inject_noise_batch, NoiseConfig


# ======================================================================
#  Data loading (reuse from existing demo scripts)
# ======================================================================

def load_index_with_noisy_meta(cache_npz: str, noise_cfg: NoiseConfig,
                                 seed: int = 42):
    rng = random.Random(seed)
    data = np.load(cache_npz, allow_pickle=True)
    vids = [str(x) for x in data["video_ids"]]
    protos_all = data["protos"].astype(np.float32)
    pcs = data["proto_counts"].astype(np.int32)
    pa = protos_all / (np.linalg.norm(protos_all, axis=-1, keepdims=True) + 1e-9)

    from datetime import datetime, timezone
    t_min = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
    t_max = datetime(2026, 4, 20, tzinfo=timezone.utc).timestamp()
    GEO_POOL = ["coast", "mountain", "urban", "indoor_home", "rural",
                 "unknown", "unknown", "unknown"]
    MOT_POOL = ["running", "walking", "stationary", "stationary",
                 "vehicle", "unknown"]

    entries = []
    clean_metas = []
    for i, vid in enumerate(vids):
        p6 = pa[i]
        p4 = np.stack([p6[:2].mean(0), p6[2:3].mean(0),
                        p6[3:5].mean(0), p6[5:6].mean(0)], axis=0)
        p4 /= np.linalg.norm(p4, axis=-1, keepdims=True) + 1e-9
        p2 = np.stack([p6[:3].mean(0), p6[3:].mean(0)], axis=0)
        p2 /= np.linalg.norm(p2, axis=-1, keepdims=True) + 1e-9

        m = VideoMetadata(
            creation_time=rng.uniform(t_min, t_max),
            geo_category=rng.choice(GEO_POOL),
            motion_class=rng.choice(MOT_POOL),
            motion_confidence=rng.uniform(0.5, 1.0),
        )
        clean_metas.append(m)
        entries.append(VideoIndexEntry(
            video_id=vid, video_path="", duration=0.0, key_ts=[],
            frame_embs=p6, protos={2: p2, 4: p4, 6: p6},
            metadata=m,
        ))

    # Inject noise
    noisy_metas = inject_noise_batch(clean_metas, noise_cfg, seed=seed)
    for i, nm in enumerate(noisy_metas):
        entries[i].metadata = nm

    return OfflineIndex(entries=entries), vids, clean_metas


def load_queries(csv_path: str):
    qs, gt = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = __import__("csv").DictReader(f)
        for row in reader:
            qs.append(row["sentence"])
            gt.append(row["video_id"])
    return qs, gt


def make_intents(queries, gt, clean_metas, vids, rng, time_prob=0.5, meta_prob=0.5):
    vid_to_meta = dict(zip(vids, clean_metas))
    parser = QueryParser()
    intents = []
    for i, (q, g) in enumerate(zip(queries, gt)):
        it = parser.parse(q)
        m = vid_to_meta.get(g)
        if m is not None:
            if m.creation_time and rng.random() < time_prob:
                half = 14 * 86400
                it.time_window = (m.creation_time - half, m.creation_time + half)
            if rng.random() < meta_prob:
                if m.geo_category and m.geo_category != "unknown":
                    it.geo_categories = [m.geo_category]
                if m.motion_class and m.motion_class != "unknown":
                    it.motion_classes = [m.motion_class]
        intents.append(it)
    return intents


# ======================================================================
#  Main pipeline
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-embs", required=True)
    ap.add_argument("--out-dir", default="reports/aaai_main")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--noise-time-std", type=float, default=7.0)
    ap.add_argument("--noise-geo-missing", type=float, default=0.3)
    ap.add_argument("--noise-geo-wrong", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    noise_cfg = NoiseConfig(
        time_shift_days_std=args.noise_time_std,
        geo_missing_prob=args.noise_geo_missing,
        geo_wrong_region_prob=args.noise_geo_wrong,
    )

    # ── Step 1: Load data ──
    print("[1/7] Loading data ...")
    index, vids, clean_metas = load_index_with_noisy_meta(
        args.cache, noise_cfg, seed=args.seed
    )
    queries, gt = load_queries(args.csv)
    q_embs = np.load(args.text_embs).astype(np.float32)[:len(queries)]
    q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9
    intents = make_intents(queries, gt, clean_metas, vids, rng)
    n_con = sum(1 for it in intents if it.has_constraint())
    print(f"   videos={index.size}  queries={len(queries)}  "
          f"constrained={n_con}/{len(queries)}")

    # ── Step 2: Build route bank labels ──
    print("[2/7] Building counterfactual route bank labels ...")
    bank = RouteBank.from_yaml()
    print(f"   routes={len(bank)}  {bank.summary()}")
    executor = RouteExecutor(index)

    # Split: 70% train, 15% cal, 15% test
    N = len(queries)
    perm = np.random.permutation(N)
    n_train = int(N * 0.70)
    n_cal = int(N * 0.15)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train + n_cal]
    test_idx = perm[n_train + n_cal:]
    print(f"   split: train={len(train_idx)} cal={len(cal_idx)} test={len(test_idx)}")

    t0 = time.perf_counter()
    labels = build_route_bank_labels(
        index, bank, q_embs[train_idx], [gt[i] for i in train_idx],
        [intents[i] for i in train_idx], MetaFilter(),
    )
    labels.save(str(out / "route_bank_train.npz"))
    print(f"   built in {time.perf_counter()-t0:.0f}s  "
          f"oracle_R@1={float((labels.ranks[np.arange(labels.n_queries), labels.oracle_route_idx]==0).mean())*100:.1f}%")

    # ── Step 3: Extract features ──
    print("[3/7] Extracting C-QIN features ...")
    meta_avail = np.array([
        sum(1 for e in index.entries if e.metadata and e.metadata.creation_time) / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.geo_category and e.metadata.geo_category != "unknown") / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.motion_class and e.metadata.motion_class != "unknown") / index.size,
        sum(1 for e in index.entries if e.metadata and e.metadata.device_make) / index.size,
    ], dtype=np.float32)

    def _feats(idx_arr):
        feats = []
        for i in idx_arr:
            sem_hits = index.search_batch(q_embs[i:i+1], top_k=20,
                                           col_beta=0.0, topm_rerank=100)[0]
            scores = np.array([s for _, s, _ in sem_hits[:20]], dtype=np.float32)
            f = extract_qin_features(
                queries[i], q_embs[i], scores, intents[i], meta_avail,
            )
            feats.append(f)
        return np.stack(feats, axis=0).astype(np.float32)

    train_feats = _feats(train_idx)
    print(f"   train features shape={train_feats.shape}")

    # ── Step 4: Train C-QIN ──
    print("[4/7] Training C-QIN ...")
    cfg = TrainConfig(epochs=args.epochs, batch_size=128, patience=30)
    model, history = train_cqin(
        train_feats, labels, cfg,
        save_dir=str(out / "model"),
    )

    # ── Step 5: Calibrate ──
    print("[5/7] Calibrating safety thresholds ...")
    cal_labels = build_route_bank_labels(
        index, bank, q_embs[cal_idx], [gt[i] for i in cal_idx],
        [intents[i] for i in cal_idx], MetaFilter(), verbose=False,
    )
    cal_feats = _feats(cal_idx)

    with __import__("torch").no_grad():
        import torch
        x = torch.from_numpy(cal_feats).float()
        cal_out = model(x)
        cal_safety = cal_out["safety_probs"].numpy()

    cal_results = calibrate_all_axes(cal_safety, cal_labels.survival_labels,
                                       delta=0.05, min_accept=10)
    save_calibration(cal_results, str(out / "calibration.json"))
    for axis, cr in cal_results.items():
        print(f"   {axis}: tau={cr.tau:.3f}  enabled={cr.enabled}  "
              f"n_acc={cr.n_accepted}  fail_ucb={cr.ucb_failure_rate:.3f}")

    # ── Step 6: Evaluate all methods on TEST split ──
    print("[6/7] Evaluating on test split ...")
    planner = CalibratedPlanner(model, bank, cal_results)
    test_executor = RouteExecutor(index)

    baselines = get_all_baselines(model=model, planner=planner)
    test_intents = [intents[i] for i in test_idx]
    test_gt = [gt[i] for i in test_idx]
    test_embs = q_embs[test_idx]

    # Oracle ranks on test set
    oracle_results = []
    for i in range(len(test_idx)):
        from routing.baselines import b4_oracle_route
        res = b4_oracle_route(test_embs[i], test_gt[i], test_intents[i],
                                test_executor, bank)
        oracle_results.append(res.rank)
    oracle_ranks = np.array(oracle_results, dtype=np.int32)

    all_results = evaluate_all_methods(
        baselines, test_embs, test_gt, test_intents,
        test_executor, bank, oracle_ranks=oracle_ranks,
    )

    # ── Step 7: Export tables ──
    print("[7/7] Exporting tables ...")
    # Main results CSV
    csv_path = out / "main_results.csv"
    if all_results:
        keys = list(all_results[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_results:
                w.writerow(r)

    # Pretty print
    print("\n" + "=" * 90)
    print(f"C-QIN AAAI Results — MSR-VTT 1K (noisy metadata, test split n={len(test_idx)})")
    print("=" * 90)
    print(f"{'Method':<28} {'R@1':>6} {'R@5':>6} {'R@10':>7} {'MeanR':>7} "
          f"{'MRR':>6} {'GT_filt':>8} {'ms/q':>7}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['method']:<28} {r['R@1']*100:>5.1f}% {r['R@5']*100:>5.1f}% "
              f"{r['R@10']*100:>6.1f}% {r['MeanR']:>7.1f} "
              f"{r['MRR']:>5.3f} {r['GT_filtered_rate']*100:>7.1f}% "
              f"{r['avg_ms_per_query']:>6.1f}")
    print("=" * 90)

    # Save full JSON
    (out / "all_results.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {out / 'all_results.json'}")
    print(f"[saved] {out / 'calibration.json'}")


if __name__ == "__main__":
    main()
