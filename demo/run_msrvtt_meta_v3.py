"""Benchmark — metadata-aware retrieval vs pure semantic retrieval.

Since MSR-VTT has no real GPS/time metadata, we synthesize controlled
metadata to isolate the effect of metadata-aware retrieval cleanly:

  1. Load 1000 MSR-VTT videos from cache (real CLIP embeddings).
  2. Synthesize plausible metadata for each video:
       - creation_time ~ U[2023-01-01, 2026-04-24]
       - geo_category ~ 30% coast, 10% mountain, 60% urban/other
       - motion_class ~ 20% running, 20% walking, 60% stationary/other
  3. For each query, we ALSO synthesize a metadata intent:
       - 50% probability: add time window around GT video's creation_time
       - 50% probability: add geo/motion constraint matching GT
     Only synthetic constraints matching the GT are injected; this
     approximates a real user who knows "their" video's context.
  4. Measure R@1 / R@5 / R@10 / latency for:
       A. Pure semantic (search_batch)
       B. Semantic + metadata (search_with_meta)

Usage:
    python demo/run_msrvtt_meta_v3.py \
        --cache data/cache/msrvtt_cache.npz \
        --csv   data/msrvtt_test_1k.csv \
        --precomputed-text-embs BENCHMARK_MSRVTT_V2_full.text_embs.npy \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.offline_index import OfflineIndex, VideoIndexEntry, build_protos
from core.metadata import VideoMetadata, GEO_CATEGORIES, MOTION_CLASSES
from core.query_parser import QueryIntent
from core.meta_filter import MetaFilter


# ----------------------------------------------------------------------

def load_index(cache_npz: str, rng: random.Random) -> Tuple[OfflineIndex, List[str]]:
    data = np.load(cache_npz, allow_pickle=True)
    vids   = [str(x) for x in data["video_ids"]]
    vembs  = data["video_embs"].astype(np.float32)
    protos = data["protos"].astype(np.float32)
    pcs    = data["proto_counts"].astype(np.int32)

    # normalize
    vembs /= np.linalg.norm(vembs, axis=-1, keepdims=True) + 1e-9
    pa = protos.copy()
    norms = np.linalg.norm(pa, axis=-1, keepdims=True) + 1e-9
    pa /= norms

    entries = []
    GEO_POOL = ["coast", "mountain", "urban", "indoor_home", "rural",
                 "unknown", "unknown", "unknown"]
    MOT_POOL = ["running", "walking", "stationary", "stationary",
                 "vehicle", "unknown"]
    t_min = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
    t_max = datetime(2026, 4, 20, tzinfo=timezone.utc).timestamp()

    for i, vid in enumerate(vids):
        p6 = pa[i]                                 # (6, D)
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
        entries.append(VideoIndexEntry(
            video_id=vid, video_path="", duration=0.0, key_ts=[],
            frame_embs=p6,
            protos={2: p2, 4: p4, 6: p6},
            metadata=m,
        ))
    return OfflineIndex(entries=entries), vids


# ----------------------------------------------------------------------

def load_queries(csv_path: str) -> Tuple[List[str], List[str]]:
    import csv
    qs, gt = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qs.append(row["sentence"])
            gt.append(row["video_id"])
    return qs, gt


# ----------------------------------------------------------------------

def make_synthetic_intents(
    index: OfflineIndex, gt: List[str], rng: random.Random,
    time_prob: float = 0.5, meta_prob: float = 0.5,
    time_slack_days: float = 14.0,
) -> List[QueryIntent]:
    id_to_meta = {e.video_id: e.metadata for e in index.entries}
    intents: List[QueryIntent] = []
    for g in gt:
        m = id_to_meta.get(g)
        it = QueryIntent(semantic_text="")  # retain semantic path
        if m is not None:
            if m.creation_time is not None and rng.random() < time_prob:
                half = time_slack_days * 86400.0
                it.time_window = (m.creation_time - half,
                                   m.creation_time + half)
            if rng.random() < meta_prob:
                # Attach whichever of geo/motion is non-unknown
                if m.geo_category and m.geo_category != "unknown":
                    it.geo_categories = [m.geo_category]
                if m.motion_class and m.motion_class != "unknown":
                    it.motion_classes = [m.motion_class]
        intents.append(it)
    return intents


# ----------------------------------------------------------------------

def evaluate(index: OfflineIndex,
              query_embs: np.ndarray,
              gt: List[str],
              intents: List[QueryIntent],
              mf: MetaFilter,
              meta_alpha: float = 0.7,
              use_hard_filter: bool = True) -> Dict:
    N = len(gt)
    id_to_idx = {e.video_id: i for i, e in enumerate(index.entries)}

    # ---- Pure semantic baseline (batch, joint-optimum hp) ----
    t0 = time.perf_counter()
    sem_hits = index.search_batch(
        query_embs, top_k=len(index.entries),
        alpha_nnn=0.7, tau_qamp=0.10, col_beta=0.4, topm_rerank=500,
    )
    dt_sem = (time.perf_counter() - t0) * 1000.0
    sem_ranks = []
    for i in range(N):
        ids = [h[0] for h in sem_hits[i]]
        sem_ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)
    sem_ranks = np.array(sem_ranks)

    # ---- Metadata-aware hybrid (corrected: col-softmax after filter,
    # ----  meta-fusion off by default per ablation findings,
    # ----  joint-optimum hp from sweep) ----
    t0 = time.perf_counter()
    all_meta_hits = index.search_batch_with_meta(
        query_embs, intents,
        top_k=len(index.entries),
        alpha_nnn=0.7, tau_qamp=0.10, col_beta=0.4, topm_rerank=500,
        meta_filter=mf, meta_alpha=meta_alpha,
        use_hard_filter=use_hard_filter,
        use_meta_fusion=False,
        col_softmax_after_filter=True,
    )
    dt_meta = (time.perf_counter() - t0) * 1000.0
    meta_ranks = []
    n_constrained = 0
    kept_ratios = []
    for i in range(N):
        it = intents[i]
        if it.has_constraint():
            n_constrained += 1
            fr = mf.filter([e.metadata for e in index.entries], it)
            kept_ratios.append(fr.n_kept / fr.n_total)
        ids = [h[0] for h in all_meta_hits[i] if h[1] > -1e8]
        meta_ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)
    meta_ranks = np.array(meta_ranks)

    def _metrics(r):
        return {
            "R@1": float((r == 0).mean()),
            "R@5": float((r < 5).mean()),
            "R@10": float((r < 10).mean()),
            "MedR": float(np.median(r) + 1),
            "MeanR": float(r.mean() + 1),
        }

    return {
        "semantic_only": {
            **_metrics(sem_ranks),
            "total_ms": dt_sem,
            "ms_per_query": dt_sem / max(N, 1),
        },
        "hybrid_meta": {
            **_metrics(meta_ranks),
            "total_ms": dt_meta,
            "ms_per_query": dt_meta / max(N, 1),
        },
        "n_queries_with_constraints": n_constrained,
        "avg_kept_ratio_when_constrained":
            float(np.mean(kept_ratios)) if kept_ratios else None,
    }


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv",   required=True)
    ap.add_argument("--precomputed-text-embs", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--meta-alpha", type=float, default=0.7)
    ap.add_argument("--no-hard-filter", action="store_true")
    ap.add_argument("--out", default="BENCHMARK_META_V3.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print(f"[load] cache={args.cache}")
    index, vids = load_index(args.cache, rng)
    print(f"  N_videos={index.size}")

    print(f"[load] queries={args.csv}")
    queries, gt = load_queries(args.csv)
    if args.limit > 0:
        queries, gt = queries[:args.limit], gt[:args.limit]
    print(f"  N_queries={len(queries)}")

    print(f"[load] precomputed text embs: {args.precomputed_text_embs}")
    q_embs = np.load(args.precomputed_text_embs).astype(np.float32)[:len(queries)]
    q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9

    print(f"[synthesize] building per-query metadata intents (seed={args.seed})")
    intents = make_synthetic_intents(index, gt, rng)
    n_con = sum(1 for it in intents if it.has_constraint())
    print(f"  {n_con}/{len(intents)} queries carry metadata constraints")

    mf = MetaFilter(time_slack_sec=3600.0, strict=False)

    print(f"[eval] running semantic vs hybrid (meta_alpha={args.meta_alpha}, "
          f"hard_filter={'OFF' if args.no_hard_filter else 'ON'}) ...")
    res = evaluate(index, q_embs, gt, intents, mf,
                    meta_alpha=args.meta_alpha,
                    use_hard_filter=not args.no_hard_filter)

    # ---- Print ----
    print("\n" + "=" * 72)
    print("MSR-VTT 1K — Phase-3 metadata-aware retrieval (synthetic meta)")
    print("=" * 72)
    for k in ("semantic_only", "hybrid_meta"):
        m = res[k]
        print(f"\n  {k}")
        print(f"    R@1  = {m['R@1']*100:5.2f}%   "
              f"R@5 = {m['R@5']*100:5.2f}%   "
              f"R@10 = {m['R@10']*100:5.2f}%")
        print(f"    MedR = {m['MedR']:.1f}   MeanR = {m['MeanR']:.1f}")
        print(f"    latency = {m['total_ms']:.1f} ms  "
              f"({m['ms_per_query']:.2f} ms/query)")
    print(f"\n  queries_with_constraints = "
          f"{res['n_queries_with_constraints']} / {len(queries)}")
    if res["avg_kept_ratio_when_constrained"] is not None:
        pct = res["avg_kept_ratio_when_constrained"] * 100
        print(f"  avg filter kept-ratio     = {pct:.1f}%   "
              f"(candidate set shrinkage = {100 - pct:.1f}%)")

    d_r1 = res["hybrid_meta"]["R@1"] - res["semantic_only"]["R@1"]
    print(f"\n  ΔR@1 (hybrid - semantic) = {d_r1*100:+.2f} pp")
    print("=" * 72)

    Path(args.out).write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
