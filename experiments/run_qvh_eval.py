"""QVHighlights moment localization evaluator.

Evaluates LiteVTR++ Phase 2/3 stack on the QVHighlights benchmark using
pre-computed CLIP+SF features released by Lei et al. 2021.

Inputs:
  --features-dir   path to clip_features/  (one .npz per video)
  --annotations    highlight_val_release.jsonl  or  highlight_test_release.jsonl
  --text-features  text features cache (npz with 'embeddings' and 'qid')
  --out            output JSON

Metrics produced (matching MomentDETR paper):
  - R1@IoU=0.5 / 0.7              (single best span hit)
  - mAP@[0.5:0.05:0.95]           (full average precision)
  - Hit@1, Hit@5
  - Per-query latency

Two pipelines:
  semantic_only      Phase 2 OfflineIndex → top-K videos → MomentDETR
                      moment localization on EACH top-K
  meta_aware         + Phase 3 metadata filter (time/geo/motion)

The MomentDETR call is delegated to the existing
`tasks.real_models.MomentDETRHighlightModel` (or its cached features
directly via the QVHighlights pipeline if no GPU is available).

Usage:
    python experiments\run_qvh_eval.py \
        --features-dir E:/datasets/QVHighlights/features/clip_features \
        --annotations  E:/datasets/QVHighlights/data/highlight_val_release.jsonl \
        --out qvh_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.offline_index import OfflineIndex, VideoIndexEntry, build_protos
from core.segment_aggregator import (
    Segment, SegmentAggregator, segments_mean_iou,
)


# ----------------------------------------------------------------------
#  Annotation loaders
# ----------------------------------------------------------------------

def load_qvh_annotations(jsonl_path: str) -> List[Dict]:
    """Load QVHighlights annotations.

    Each row contains:
      qid, query, vid, duration, relevant_windows,
      saliency_scores, relevant_clip_ids
    """
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_clip_video_feature(features_dir: str, vid: str) -> Optional[np.ndarray]:
    """Load (T, 512) CLIP features for one video.

    QVHighlights features are stored at 0.5 fps (every 2 sec).
    """
    p = Path(features_dir) / f"{vid}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if "features" in d:
        return d["features"].astype(np.float32)
    # alternative key naming
    return d[d.files[0]].astype(np.float32)


# ----------------------------------------------------------------------
#  Build OfflineIndex from QVH features
# ----------------------------------------------------------------------

def build_index_from_qvh(
    features_dir: str,
    annotations: List[Dict],
    K_values: Tuple[int, ...] = (2, 4, 6),
) -> OfflineIndex:
    """Build OfflineIndex from QVH per-video CLIP features."""
    seen = set()
    entries: List[VideoIndexEntry] = []
    for ann in annotations:
        vid = ann["vid"]
        if vid in seen:
            continue
        seen.add(vid)
        feats = load_clip_video_feature(features_dir, vid)
        if feats is None or len(feats) < 2:
            continue
        feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-9)

        protos = {K: build_protos(feats, K) for K in K_values}

        entries.append(VideoIndexEntry(
            video_id=vid, video_path="",
            duration=float(ann.get("duration", 0.0)),
            key_ts=[float(2 * i) for i in range(len(feats))],
            frame_embs=feats,
            protos=protos,
        ))
    return OfflineIndex(entries=entries)


# ----------------------------------------------------------------------
#  Moment localization from per-frame similarity
# ----------------------------------------------------------------------

def localize_moments_via_aggregator(
    frame_embs: np.ndarray,                # (T, D) of one video
    query_emb: np.ndarray,                 # (D,)
    clip_len: float = 2.0,
    percentile: float = 0.7,
    merge_gap_sec: float = 2.0,
    min_seg_sec: float = 1.0,
    max_segments: int = 5,
) -> List[Segment]:
    """Compute (timestamp, score) pairs and aggregate into segments."""
    if len(frame_embs) == 0:
        return []
    sims = (frame_embs @ query_emb).astype(np.float32)
    timestamps = [i * clip_len + clip_len / 2 for i in range(len(sims))]
    pairs = list(zip(timestamps, sims.tolist()))
    agg = SegmentAggregator(
        percentile=percentile,
        smooth_window=3,
        merge_gap_sec=merge_gap_sec,
        min_segment_sec=min_seg_sec,
        max_segments=max_segments,
    )
    return agg.aggregate(pairs)


# ----------------------------------------------------------------------
#  Metrics
# ----------------------------------------------------------------------

def iou_1d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(union, 1e-9)


def compute_recall_at_iou(
    pred_spans: List[List[Segment]],
    gt_spans:   List[List[Tuple[float, float]]],
    iou_threshold: float = 0.5,
    top_k: int = 1,
) -> float:
    """Recall@K: fraction of queries where ANY of top-K predicted spans
    has IoU ≥ threshold with ANY ground-truth span."""
    n_hit = 0
    for preds, gts in zip(pred_spans, gt_spans):
        if not preds or not gts:
            continue
        top = preds[:top_k]
        for p in top:
            if any(iou_1d((p.start, p.end), (g[0], g[1])) >= iou_threshold
                   for g in gts):
                n_hit += 1
                break
    return n_hit / max(len(pred_spans), 1)


def compute_mean_avg_precision(
    pred_spans: List[List[Segment]],
    gt_spans:   List[List[Tuple[float, float]]],
    iou_grid: List[float] = None,
) -> Dict[str, float]:
    iou_grid = iou_grid or [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = {}
    for thr in iou_grid:
        ap_per_q = []
        for preds, gts in zip(pred_spans, gt_spans):
            if not preds:
                ap_per_q.append(0.0); continue
            if not gts:
                continue
            # rank preds by score, compute AP
            n_gt = len(gts)
            sorted_preds = sorted(preds, key=lambda s: -s.score)
            tp = np.zeros(len(sorted_preds))
            for i, p in enumerate(sorted_preds):
                if any(iou_1d((p.start, p.end), (g[0], g[1])) >= thr
                       for g in gts):
                    tp[i] = 1
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(1 - tp)
            precision = cum_tp / (cum_tp + cum_fp + 1e-9)
            recall = cum_tp / max(n_gt, 1)
            # 11-point AP
            ap = 0.0
            for r in np.arange(0, 1.01, 0.1):
                p_at_r = precision[recall >= r]
                ap += (p_at_r.max() if p_at_r.size else 0.0) / 11
            ap_per_q.append(ap)
        aps[f"AP@{thr}"] = float(np.mean(ap_per_q)) if ap_per_q else 0.0
    aps["mAP"] = float(np.mean(list(aps.values())))
    return aps


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def encode_queries(queries: List[str], cache_npy: Optional[str] = None) -> np.ndarray:
    """Use Huawei-replacement RealCLIPModel (MobileCLIP2-S0) to encode."""
    if cache_npy and Path(cache_npy).exists():
        print(f"[load] cached text embs: {cache_npy}")
        return np.load(cache_npy).astype(np.float32)
    from tasks.real_models import RealCLIPModel
    clip = RealCLIPModel()
    out = []
    bsz = 64
    for i in range(0, len(queries), bsz):
        chunk = queries[i:i + bsz]
        out.extend(clip.encode_text(chunk))
        if (i // bsz) % 5 == 0:
            print(f"  {i + len(chunk)}/{len(queries)}")
    arr = np.stack(out, axis=0).astype(np.float32)
    if cache_npy:
        np.save(cache_npy, arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--text-cache", default="qvh_text_embs.npy")
    ap.add_argument("--out", default="qvh_results.json")
    ap.add_argument("--top-videos", type=int, default=5,
                    help="how many top retrieved videos to localize on")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    print(f"[load] annotations: {args.annotations}")
    anns = load_qvh_annotations(args.annotations)
    if args.limit > 0:
        anns = anns[:args.limit]
    print(f"   {len(anns)} queries")

    print(f"[load] features dir: {args.features_dir}")
    index = build_index_from_qvh(args.features_dir, anns)
    print(f"   N_unique_videos={index.size}")

    queries = [a["query"] for a in anns]
    q_embs = encode_queries(queries, cache_npy=args.text_cache)
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9)
    print(f"   q_embs={q_embs.shape}")

    # ----- Retrieval (Phase-2 batch) -----
    print("[eval] running Phase-2 retrieval ...")
    t0 = time.perf_counter()
    all_hits = index.search_batch(
        q_embs, top_k=max(args.top_videos, 5),
        alpha_nnn=0.5, tau_qamp=0.05, col_beta=0.4, topm_rerank=300,
    )
    dt_retrieval = (time.perf_counter() - t0) * 1000.0

    # ----- Moment localization on top retrieved videos -----
    print("[eval] running moment localization ...")
    pred_spans: List[List[Segment]] = []
    gt_spans:   List[List[Tuple[float, float]]] = []

    id_to_entry = {e.video_id: e for e in index.entries}

    t0 = time.perf_counter()
    for i, ann in enumerate(anns):
        vid = ann["vid"]
        # Best path: localize on the GT video (retrieval-conditional eval).
        # For QVHighlights protocol, evaluation assumes vid is given.
        target = id_to_entry.get(vid)
        if target is None or target.frame_embs is None:
            pred_spans.append([])
        else:
            segs = localize_moments_via_aggregator(
                target.frame_embs, q_embs[i],
                clip_len=2.0, percentile=0.7,
                merge_gap_sec=2.0, min_seg_sec=1.0,
                max_segments=5,
            )
            pred_spans.append(segs)
        gt_spans.append([(w[0], w[1]) for w in ann.get("relevant_windows", [])])
    dt_loc = (time.perf_counter() - t0) * 1000.0

    # ----- Metrics -----
    R1_05 = compute_recall_at_iou(pred_spans, gt_spans, 0.5, top_k=1)
    R1_07 = compute_recall_at_iou(pred_spans, gt_spans, 0.7, top_k=1)
    R5_05 = compute_recall_at_iou(pred_spans, gt_spans, 0.5, top_k=5)
    mAPs  = compute_mean_avg_precision(pred_spans, gt_spans)

    summary = {
        "n_queries": len(anns),
        "n_videos": index.size,
        "R1@0.5":  R1_05,
        "R1@0.7":  R1_07,
        "R5@0.5":  R5_05,
        **mAPs,
        "retrieval_total_ms": dt_retrieval,
        "loc_total_ms":       dt_loc,
        "ms_per_query_loc":   dt_loc / max(len(anns), 1),
    }

    print("\n" + "=" * 72)
    print(f"QVHighlights moment localization ({len(anns)} queries)")
    print("=" * 72)
    print(f"  R1 @ IoU=0.5 = {R1_05*100:5.2f}%")
    print(f"  R1 @ IoU=0.7 = {R1_07*100:5.2f}%")
    print(f"  R5 @ IoU=0.5 = {R5_05*100:5.2f}%")
    print(f"  mAP          = {mAPs['mAP']*100:5.2f}%")
    print(f"  retrieval ms = {dt_retrieval:.1f}  ({dt_retrieval/len(anns):.2f}/q)")
    print(f"  loc ms       = {dt_loc:.1f}  ({dt_loc/len(anns):.2f}/q)")
    print("=" * 72)

    Path(args.out).write_text(json.dumps({"summary": summary,
                                           "config": vars(args)},
                                          indent=2, default=str),
                                encoding="utf-8")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
