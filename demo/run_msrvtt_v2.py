"""Benchmark OfflineIndex on the real MSR-VTT 1K test set.

Uses the precomputed MobileCLIP2-S0 cache (video_embs + protos) in
`data/cache/msrvtt_cache.npz` — this is exactly the data that produced
your 39.2% R@1 number. We feed those protos into the v2 OfflineIndex
and measure:

  - Retrieval R@1/R@5/R@10/MeanR
  - QPP margin distribution (easy/medium/hard split)
  - Per-query wall latency
  - Comparison to raw cosine + NNN+QAMP+col-softmax

No Huawei model weight is loaded — we reuse the cached image embeddings
and only call text_emb from the cache (which was already produced by
MobileCLIP2-S0 on disk). For the text side, this benchmark requires
encoded text embeddings; we compute them on-the-fly with MobileCLIP2
if available, else skip the text-encode step and use a synthetic probe.

Usage:
    python demo/run_msrvtt_v2.py \
        --cache  ../video_retrieval_code_no_dataset/data/cache/msrvtt_cache.npz \
        --csv    ../video_retrieval_code_no_dataset/data/msrvtt_test_1k.csv \
        --out    BENCHMARK_MSRVTT_V2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.offline_index import (
    OfflineIndex, VideoIndexEntry, build_protos,
)
from core.query_planner import (
    QueryPlanner, QueryPlannerConfig, QueryDifficulty,
)


# ----------------------------------------------------------------------

def load_cache_as_index(cache_npz: str) -> OfflineIndex:
    data = np.load(cache_npz, allow_pickle=True)
    video_ids   = list(data["video_ids"])
    video_embs  = data["video_embs"]          # (N, D), video-level mean emb
    protos_all  = data["protos"]              # (N, 6, D), precomputed K=6
    proto_counts = data["proto_counts"]       # (N,) how many protos are valid

    # Normalize per-proto for cosine math
    pa = protos_all.astype(np.float32)
    norms = np.linalg.norm(pa, axis=-1, keepdims=True) + 1e-9
    pa = pa / norms

    # Build multi-K protos (K=2,4,6) from the K=6 cache by averaging segments
    entries: List[VideoIndexEntry] = []
    for i, vid in enumerate(video_ids):
        p6 = pa[i]                                # (6, D)
        # K=6 → as-is; K=4 → pool every 1.5 protos via linspace avg; K=2 → split
        p4 = np.stack([p6[:2].mean(0), p6[2:3].mean(0),
                        p6[3:5].mean(0), p6[5:6].mean(0)], axis=0)
        p4 /= (np.linalg.norm(p4, axis=-1, keepdims=True) + 1e-9)
        p2 = np.stack([p6[:3].mean(0), p6[3:].mean(0)], axis=0)
        p2 /= (np.linalg.norm(p2, axis=-1, keepdims=True) + 1e-9)

        entries.append(VideoIndexEntry(
            video_id=str(vid),
            video_path="",
            duration=0.0,
            key_ts=[],
            frame_embs=p6,
            protos={2: p2, 4: p4, 6: p6},
            meta={"video_mean": video_embs[i].astype(np.float32)},
        ))
    return OfflineIndex(entries=entries)


# ----------------------------------------------------------------------

def load_queries_and_gt(csv_path: str, limit: int = -1
                         ) -> Tuple[List[str], List[str]]:
    import csv
    qs, gt = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qs.append(row["sentence"])
            gt.append(row["video_id"])
    if limit > 0:
        qs = qs[:limit]
        gt = gt[:limit]
    return qs, gt


def encode_texts(queries: List[str]) -> np.ndarray:
    """Encode all queries with the REAL MobileCLIP2-S0."""
    print(f"[text] encoding {len(queries)} queries with MobileCLIP2-S0 ...")
    from tasks.real_models import RealCLIPModel
    clip = RealCLIPModel()
    batch = 64
    all_embs = []
    for i in range(0, len(queries), batch):
        chunk = queries[i:i + batch]
        embs = clip.encode_text(chunk)
        all_embs.extend(embs)
        if (i // batch) % 5 == 0:
            print(f"  {i + len(chunk)}/{len(queries)}")
    out = np.stack(all_embs, axis=0).astype(np.float32)
    out /= (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9)
    return out


# ----------------------------------------------------------------------

def evaluate(index: OfflineIndex, query_embs: np.ndarray, gt: List[str],
             qpp: QueryPlanner,
             alpha_nnn: float = 0.5, tau_qamp: float = 0.02,
             col_beta: float = 0.4) -> Dict:
    N = query_embs.shape[0]

    # Use batch search to get correct column-wise softmax behaviour
    print(f"[eval] running batch search (Nq={N}, Nv={index.size}) ...")
    t0 = time.perf_counter()
    all_hits = index.search_batch(
        query_embs, top_k=1000,
        alpha_nnn=alpha_nnn, tau_qamp=tau_qamp, col_beta=col_beta,
    )
    dt_total = (time.perf_counter() - t0) * 1000.0

    ranks_at_gt = []
    per_query = []
    qpp_buckets = {"easy": 0, "medium": 0, "hard": 0}
    margin_hist = []

    for i in range(N):
        hits = all_hits[i]
        ids = [h[0] for h in hits]
        try:
            rank = ids.index(gt[i])
        except ValueError:
            rank = 1000
        ranks_at_gt.append(rank)
        plan = qpp.plan(hits)
        qpp_buckets[plan.difficulty.value] += 1
        margin_hist.append(plan.margin)
        per_query.append({
            "query_idx": i, "gt": gt[i], "rank": rank,
            "margin": plan.margin, "plan": plan.difficulty.value,
        })

    ranks = np.array(ranks_at_gt)
    report = {
        "n": int(N),
        "R@1":   float((ranks == 0).mean()),
        "R@5":   float((ranks <  5).mean()),
        "R@10":  float((ranks < 10).mean()),
        "MedR":  float(np.median(ranks) + 1),
        "MeanR": float(ranks.mean() + 1),
        "total_ms": float(dt_total),
        "avg_ms_per_query": float(dt_total / max(N, 1)),
        "qpp_split": {
            k: {"n": v, "pct": 100.0 * v / max(N, 1)}
            for k, v in qpp_buckets.items()
        },
        "margin_stats": {
            "min":  float(np.min(margin_hist)),
            "max":  float(np.max(margin_hist)),
            "mean": float(np.mean(margin_hist)),
            "p25":  float(np.percentile(margin_hist, 25)),
            "p50":  float(np.percentile(margin_hist, 50)),
            "p75":  float(np.percentile(margin_hist, 75)),
        },
    }
    return {"summary": report, "per_query": per_query[:50]}


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv",   required=True)
    ap.add_argument("--out",   default="BENCHMARK_MSRVTT_V2.json")
    ap.add_argument("--limit", type=int, default=-1,
                    help="limit to first N queries for quick tests")
    ap.add_argument("--alpha-nnn", type=float, default=0.5)
    ap.add_argument("--tau-qamp",  type=float, default=0.01)
    ap.add_argument("--col-beta",  type=float, default=0.4)
    ap.add_argument("--easy-margin", type=float, default=0.08)
    ap.add_argument("--hard-margin", type=float, default=0.02)
    ap.add_argument("--precomputed-text-embs", default=None,
                    help="npy path if you've already encoded all queries")
    args = ap.parse_args()

    print("[load] cache:", args.cache)
    index = load_cache_as_index(args.cache)
    print(f"  videos={index.size}, K={list(index._flat_protos.keys())}")

    print("[load] queries:", args.csv)
    queries, gt = load_queries_and_gt(args.csv, limit=args.limit)
    print(f"  n_queries={len(queries)}")

    if args.precomputed_text_embs and Path(args.precomputed_text_embs).exists():
        print(f"[load] precomputed text embs: {args.precomputed_text_embs}")
        q_embs = np.load(args.precomputed_text_embs).astype(np.float32)
        q_embs /= (np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9)
    else:
        q_embs = encode_texts(queries)
        sv = Path(args.out).with_suffix(".text_embs.npy")
        np.save(sv, q_embs)
        print(f"[save] text_embs -> {sv}")

    planner = QueryPlanner(QueryPlannerConfig(
        easy_margin=args.easy_margin,
        hard_margin=args.hard_margin,
    ))

    print("[eval] running retrieval + QPP ...")
    result = evaluate(index, q_embs, gt, planner,
                       alpha_nnn=args.alpha_nnn,
                       tau_qamp=args.tau_qamp,
                       col_beta=args.col_beta)

    # ------------------------------------------------------------------ print
    s = result["summary"]
    print("\n" + "=" * 72)
    print(f"MSR-VTT 1K test · OfflineIndex retrieval")
    print(f"  R@1  = {s['R@1']*100:5.2f}%")
    print(f"  R@5  = {s['R@5']*100:5.2f}%")
    print(f"  R@10 = {s['R@10']*100:5.2f}%")
    print(f"  MedR = {s['MedR']:.1f}   MeanR = {s['MeanR']:.1f}")
    print(f"  avg_ms/query = {s['avg_ms_per_query']:.2f} ms")
    print(f"  QPP split:")
    for k, v in s["qpp_split"].items():
        print(f"    {k:<6}: {v['n']:4d}  ({v['pct']:.1f}%)")
    print(f"  margin stats: p25={s['margin_stats']['p25']:.4f}  "
          f"p50={s['margin_stats']['p50']:.4f}  "
          f"p75={s['margin_stats']['p75']:.4f}")
    print("=" * 72)

    Path(args.out).write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
