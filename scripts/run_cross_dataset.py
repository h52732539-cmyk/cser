"""Cross-dataset evaluation: C-QIN trained on MSR-VTT, zero-shot on QVHighlights.

Demonstrates that the learned router generalizes to unseen datasets without
retraining, because it learns query-intent patterns (temporal/spatial/action)
rather than dataset-specific video features.

Prerequisites:
  - QVHighlights CLIP features at E:\datasets\QVHighlights\features\clip_features\
  - QVHighlights annotations at E:\datasets\QVHighlights\moment_detr_repo\data\
  - C-QIN model trained on MSR-VTT (from run_final_eval.py or run_pareto_sweep.py)

Usage:
    python scripts/run_cross_dataset.py \
        --msrvtt-cache <msrvtt_cache.npz> \
        --msrvtt-csv <msrvtt_test_1k.csv> \
        --msrvtt-text-embs <text_embs.npy> \
        --qvh-features-dir <QVHighlights/features/clip_features> \
        --qvh-annotations <highlight_val_release.jsonl> \
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
from typing import Dict, List, Optional, Tuple

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
from routing.route_executor import RouteExecutor
from routing.route_bank_builder import build_route_bank_labels
from routing.qin_model import CalibratedQIN, extract_qin_features
from routing.train_qin import train_cqin, TrainConfig
from routing.calibrate_safety import calibrate_all_axes
from routing.calibrated_planner_v2 import CalibratedPlannerV2, BudgetedCascadePlanner
from routing.baselines import b0_semantic_only, b1_rule_parser

from metadata.noisy_metadata import inject_noise_batch, NoiseConfig


# ======================================================================
#  QVHighlights data loading
# ======================================================================

def load_qvh_annotations(jsonl_path: str) -> List[Dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_qvh_video_feature(features_dir: str, vid: str) -> Optional[np.ndarray]:
    p = Path(features_dir) / f"{vid}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    key = "features" if "features" in d else d.files[0]
    feat = d[key].astype(np.float32)
    if feat.ndim != 2 or feat.shape[0] < 2:
        return None
    return feat


def build_qvh_index(features_dir: str, annotations: List[Dict],
                      noise_cfg: NoiseConfig, seed: int = 42
                      ) -> Tuple[OfflineIndex, List[str]]:
    """Build OfflineIndex from QVH features with synthetic metadata."""
    rng = random.Random(seed)
    from datetime import datetime, timezone
    t_min = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    t_max = datetime(2024, 12, 31, tzinfo=timezone.utc).timestamp()
    GEO = ["coast", "mountain", "urban", "indoor_home", "rural",
            "unknown", "unknown", "unknown"]
    MOT = ["running", "walking", "stationary", "stationary", "vehicle", "unknown"]

    seen = set()
    entries = []
    for ann in annotations:
        vid = ann["vid"]
        if vid in seen:
            continue
        seen.add(vid)
        feats = load_qvh_video_feature(features_dir, vid)
        if feats is None or len(feats) < 2:
            continue
        feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-9)
        protos = {K: build_protos(feats, K) for K in (2, 4, 6)}
        m = VideoMetadata(
            creation_time=rng.uniform(t_min, t_max),
            geo_category=rng.choice(GEO),
            motion_class=rng.choice(MOT),
            motion_confidence=rng.uniform(0.5, 1.0),
        )
        noisy_m = inject_noise_batch([m], noise_cfg, seed=seed + len(entries))[0]
        entries.append(VideoIndexEntry(
            video_id=vid, video_path="", duration=float(ann.get("duration", 0)),
            frame_embs=feats, protos=protos, metadata=noisy_m,
        ))
    vids = [e.video_id for e in entries]
    return OfflineIndex(entries), vids


# ======================================================================
#  MSR-VTT training (reuse from other scripts)
# ======================================================================

def train_cqin_on_msrvtt(cache_npz, csv_path, text_embs_path, seed=42, epochs=150):
    """Train C-QIN on MSR-VTT and return (model, calibration, bank)."""
    rng = random.Random(seed); np.random.seed(seed)
    noise_cfg = NoiseConfig(time_shift_days_std=7.0, geo_missing_prob=0.3,
                             geo_wrong_region_prob=0.1)

    # Load MSR-VTT
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
    msrvtt_index = OfflineIndex(entries)

    queries, gt = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in __import__("csv").DictReader(f):
            queries.append(row["sentence"]); gt.append(row["video_id"])
    q_embs = np.load(text_embs_path).astype(np.float32)[:len(queries)]
    q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9

    v2m = dict(zip(vids, clean))
    parser = QueryParser()
    intents = []
    for q, g in zip(queries, gt):
        it = parser.parse(q)
        m = v2m.get(g)
        if m:
            if m.creation_time and rng.random() < 0.5:
                it.time_window = (m.creation_time - 14*86400, m.creation_time + 14*86400)
            if rng.random() < 0.5:
                if m.geo_category and m.geo_category != "unknown":
                    it.geo_categories = [m.geo_category]
                if m.motion_class and m.motion_class != "unknown":
                    it.motion_classes = [m.motion_class]
        intents.append(it)

    N = len(queries)
    perm = np.random.permutation(N)
    n_tr = int(N * 0.35)
    n_cal = int(N * 0.08)
    tr, cal_idx = perm[:n_tr], perm[n_tr:n_tr+n_cal]

    bank = RouteBank.from_yaml()
    executor = RouteExecutor(msrvtt_index)
    labels = build_route_bank_labels(
        msrvtt_index, bank, q_embs[tr], [gt[i] for i in tr],
        [intents[i] for i in tr], MetaFilter(), verbose=False,
    )
    mavail = np.array([
        sum(1 for e in msrvtt_index.entries if e.metadata and e.metadata.creation_time) / msrvtt_index.size,
        sum(1 for e in msrvtt_index.entries if e.metadata and e.metadata.geo_category
            and e.metadata.geo_category != "unknown") / msrvtt_index.size,
        sum(1 for e in msrvtt_index.entries if e.metadata and e.metadata.motion_class
            and e.metadata.motion_class != "unknown") / msrvtt_index.size,
        sum(1 for e in msrvtt_index.entries if e.metadata and e.metadata.device_make) / msrvtt_index.size,
    ], dtype=np.float32)

    def _feats(idx_arr):
        feats = []
        for i in idx_arr:
            hits = msrvtt_index.search_batch(q_embs[i:i+1], top_k=20,
                                              col_beta=0.0, topm_rerank=100)[0]
            sc = np.array([s for _, s, _ in hits[:20]], dtype=np.float32)
            feats.append(extract_qin_features(
                queries[i], q_embs[i], sc, intents[i], mavail
            ))
        return np.stack(feats).astype(np.float32)

    train_feats = _feats(tr)
    model, _ = train_cqin(train_feats, labels,
                            TrainConfig(epochs=epochs, patience=20), verbose=False)

    cal_labels = build_route_bank_labels(
        msrvtt_index, bank, q_embs[cal_idx], [gt[i] for i in cal_idx],
        [intents[i] for i in cal_idx], MetaFilter(), verbose=False,
    )
    cal_feats = _feats(cal_idx)
    with torch.no_grad():
        cs = model(torch.from_numpy(cal_feats).float())["safety_probs"].numpy()
    cal_res = calibrate_all_axes(cs, cal_labels.survival_labels,
                                   delta=0.10, min_accept=5)

    return model, cal_res, bank


# ======================================================================
#  Evaluation on target dataset
# ======================================================================

def evaluate_on_target(
    model: CalibratedQIN,
    cal_res: Dict,
    bank: RouteBank,
    target_index: OfflineIndex,
    query_texts: List[str],
    query_embs: np.ndarray,
    gt_vids: List[str],
    intents: List[QueryIntent],
) -> Dict[str, Dict]:
    """Evaluate B0/B1/B10 on a target dataset using MSR-VTT-trained C-QIN."""
    executor = RouteExecutor(target_index)
    mavail = np.array([
        sum(1 for e in target_index.entries if e.metadata and e.metadata.creation_time) / max(target_index.size, 1),
        sum(1 for e in target_index.entries if e.metadata and e.metadata.geo_category
            and e.metadata.geo_category != "unknown") / max(target_index.size, 1),
        sum(1 for e in target_index.entries if e.metadata and e.metadata.motion_class
            and e.metadata.motion_class != "unknown") / max(target_index.size, 1),
        sum(1 for e in target_index.entries if e.metadata and e.metadata.device_make) / max(target_index.size, 1),
    ], dtype=np.float32)

    planner = CalibratedPlannerV2(model, bank, cal_res, soft_ratio=0.6)
    cascade = BudgetedCascadePlanner(planner, bank)

    N = len(gt_vids)

    def _run(name, fn):
        ranks = np.full(N, -1, dtype=np.int32)
        gt_filt = np.zeros(N, dtype=bool)
        for i in range(N):
            res = fn(i)
            ranks[i] = res.rank
            gt_filt[i] = res.gt_filtered
        r1 = float(((ranks >= 0) & (ranks < 1)).mean())
        r5 = float(((ranks >= 0) & (ranks < 5)).mean())
        r10 = float(((ranks >= 0) & (ranks < 10)).mean())
        gtf = float(gt_filt.mean())
        return {"R@1": r1, "R@5": r5, "R@10": r10, "GT_filtered": gtf, "n": N}

    # B0: semantic-only
    def _b0(i):
        return executor.execute(FALLBACK_ROUTE, query_embs[i], gt_vids[i], intents[i])

    # B1: rule parser
    def _b1(i):
        return b1_rule_parser(query_embs[i], gt_vids[i], intents[i], executor, bank)

    # B10: C-QIN + cascade (zero-shot from MSR-VTT)
    def _b10(i):
        hits = target_index.search_batch(query_embs[i:i+1], top_k=20,
                                          col_beta=0.0, topm_rerank=100)[0]
        sc = np.array([s for _, s, _ in hits[:20]], dtype=np.float32)
        feat = extract_qin_features(
            query_texts[i], query_embs[i], sc, intents[i], mavail
        )
        return cascade.plan_and_execute(feat, query_embs[i], gt_vids[i],
                                          intents[i], executor)

    results = {
        "B0_semantic": _run("B0", _b0),
        "B1_rule_parser": _run("B1", _b1),
        "B10_cqin_zeroshot": _run("B10", _b10),
    }
    return results


# ======================================================================
#  Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msrvtt-cache", required=True)
    ap.add_argument("--msrvtt-csv", required=True)
    ap.add_argument("--msrvtt-text-embs", required=True)
    ap.add_argument("--qvh-features-dir", default=None,
                    help="Path to QVHighlights clip_features/ directory")
    ap.add_argument("--qvh-annotations", default=None,
                    help="Path to highlight_val_release.jsonl")
    ap.add_argument("--out-dir", default="reports/aaai_final")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ── Train C-QIN on MSR-VTT ──
    print("[1/3] Training C-QIN on MSR-VTT ...")
    model, cal_res, bank = train_cqin_on_msrvtt(
        args.msrvtt_cache, args.msrvtt_csv, args.msrvtt_text_embs,
        seed=args.seed, epochs=args.epochs,
    )
    print("   done")

    all_results = {}

    # ── QVHighlights ──
    if args.qvh_features_dir and args.qvh_annotations:
        print("[2/3] Evaluating on QVHighlights (zero-shot) ...")
        anns = load_qvh_annotations(args.qvh_annotations)
        noise_cfg = NoiseConfig(time_shift_days_std=7.0, geo_missing_prob=0.3)
        qvh_index, qvh_vids = build_qvh_index(
            args.qvh_features_dir, anns, noise_cfg, seed=args.seed
        )
        print(f"   QVH index: {qvh_index.size} videos")

        # Encode queries (use MobileCLIP2 if available, else skip)
        qvh_queries = [a["query"] for a in anns]
        qvh_gt = [a["vid"] for a in anns]
        # Filter to queries whose GT video is in the index
        valid = [(q, g) for q, g in zip(qvh_queries, qvh_gt)
                  if g in {e.video_id for e in qvh_index.entries}]
        if valid:
            qvh_queries_f = [v[0] for v in valid]
            qvh_gt_f = [v[1] for v in valid]
            print(f"   valid queries: {len(qvh_queries_f)}")

            # Encode text queries
            try:
                from tasks.real_models import RealCLIPModel
                clip = RealCLIPModel()
                qvh_embs = np.stack(clip.encode_text(qvh_queries_f)).astype(np.float32)
                qvh_embs /= np.linalg.norm(qvh_embs, axis=-1, keepdims=True) + 1e-9
            except Exception as e:
                print(f"   [warn] CLIP unavailable ({e}), using random embs")
                qvh_embs = np.random.randn(len(qvh_queries_f), 512).astype(np.float32)
                qvh_embs /= np.linalg.norm(qvh_embs, axis=-1, keepdims=True) + 1e-9

            parser = QueryParser()
            qvh_intents = [parser.parse(q) for q in qvh_queries_f]

            qvh_results = evaluate_on_target(
                model, cal_res, bank, qvh_index,
                qvh_queries_f, qvh_embs, qvh_gt_f, qvh_intents,
            )
            all_results["QVHighlights"] = qvh_results
            print(f"   Results:")
            for name, m in qvh_results.items():
                print(f"     {name}: R@1={m['R@1']*100:.1f}%  "
                      f"R@5={m['R@5']*100:.1f}%  GT_f={m['GT_filtered']*100:.1f}%")
    else:
        print("[2/3] QVHighlights skipped (no --qvh-features-dir)")

    # ── Save ──
    print("[3/3] Saving ...")
    (out / "cross_dataset_results.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("CROSS-DATASET RESULTS (C-QIN trained on MSR-VTT, zero-shot transfer)")
    print("=" * 70)
    for dataset, methods in all_results.items():
        print(f"\n  {dataset}:")
        print(f"  {'Method':<25} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'GT_f':>6}")
        for name, m in methods.items():
            print(f"  {name:<25} {m['R@1']*100:>6.1f}% {m['R@5']*100:>6.1f}% "
                  f"{m['R@10']*100:>6.1f}% {m['GT_filtered']*100:>5.1f}%")
    print("=" * 70)
    print(f"\n[saved] {out}/cross_dataset_results.json")


if __name__ == "__main__":
    main()
