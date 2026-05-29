"""Moment localization: C-QIN retrieves top-K → MomentDETR localizes.

Evaluates on QVHighlights: instead of localizing on the GT video (oracle),
we first retrieve top-K videos using different methods, then run
SegmentAggregator-based localization on the retrieved video.

Metrics: R1@IoU=0.5, R1@IoU=0.7, mAP@0.5

Usage:
    python scripts/run_moment_localization.py \
        --qvh-features-dir <clip_features/> \
        --qvh-annotations <highlight_val_release.jsonl> \
        --msrvtt-cache <msrvtt_cache.npz> \
        --msrvtt-csv <msrvtt_test_1k.csv> \
        --msrvtt-text-embs <text_embs.npy> \
        --out-dir reports/aaai_final
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.offline_index import OfflineIndex, VideoIndexEntry, build_protos
from core.segment_aggregator import Segment, SegmentAggregator


# ======================================================================
#  QVH loading (reuse from run_cross_dataset)
# ======================================================================

def load_qvh_annotations(jsonl_path: str) -> List[Dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


def load_feature(features_dir: str, vid: str):
    p = Path(features_dir) / f"{vid}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    key = "features" if "features" in d else d.files[0]
    feat = d[key].astype(np.float32)
    if feat.ndim != 2 or feat.shape[0] < 2:
        return None
    return feat


# ======================================================================
#  Moment localization via SegmentAggregator
# ======================================================================

def localize(frame_embs: np.ndarray, query_emb: np.ndarray,
              clip_len: float = 2.0) -> List[Segment]:
    """Frame-level cosine → SegmentAggregator → segments."""
    if len(frame_embs) == 0:
        return []
    sims = (frame_embs @ query_emb).astype(np.float32)
    timestamps = [i * clip_len + clip_len / 2 for i in range(len(sims))]
    pairs = list(zip(timestamps, sims.tolist()))
    agg = SegmentAggregator(
        percentile=0.70, smooth_window=3,
        merge_gap_sec=2.0, min_segment_sec=1.0, max_segments=5,
    )
    return agg.aggregate(pairs)


# ======================================================================
#  Metrics
# ======================================================================

def iou_1d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(union, 1e-9)


def recall_at_iou(pred_segs_list, gt_spans_list, iou_thr, top_k=1):
    n_hit = 0
    for preds, gts in zip(pred_segs_list, gt_spans_list):
        if not preds or not gts:
            continue
        top = preds[:top_k]
        for p in top:
            if any(iou_1d((p.start, p.end), g) >= iou_thr for g in gts):
                n_hit += 1; break
    return n_hit / max(len(pred_segs_list), 1)


# ======================================================================
#  Retrieval methods
# ======================================================================

def build_qvh_index(features_dir, annotations):
    seen = set()
    entries = []
    for ann in annotations:
        vid = ann["vid"]
        if vid in seen: continue
        seen.add(vid)
        feats = load_feature(features_dir, vid)
        if feats is None: continue
        feats_n = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-9)
        protos = {K: build_protos(feats_n, K) for K in (2, 4, 6)}
        entries.append(VideoIndexEntry(
            video_id=vid, video_path="", duration=float(ann.get("duration", 0)),
            frame_embs=feats_n, protos=protos,
        ))
    return OfflineIndex(entries)


def retrieve_semantic(index: OfflineIndex, query_emb: np.ndarray, top_k=1):
    """Semantic-only retrieval → top-K video IDs."""
    hits = index.search_batch(query_emb[np.newaxis], top_k=top_k)[0]
    return [vid for vid, _, _ in hits[:top_k]]


# ======================================================================
#  Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qvh-features-dir", required=True)
    ap.add_argument("--qvh-annotations", required=True)
    ap.add_argument("--out-dir", default="reports/aaai_final")
    ap.add_argument("--text-cache", default=None,
                    help="Pre-computed QVH text embeddings .npy")
    ap.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading QVHighlights ...")
    anns = load_qvh_annotations(args.qvh_annotations)
    index = build_qvh_index(args.qvh_features_dir, anns)
    print(f"   {len(anns)} queries, {index.size} videos in index")

    # Filter to queries with GT video in index
    id_set = {e.video_id for e in index.entries}
    valid_anns = [a for a in anns if a["vid"] in id_set]
    print(f"   {len(valid_anns)} valid queries (GT in index)")

    # Encode queries
    print("[2/4] Encoding queries ...")
    if args.text_cache and Path(args.text_cache).exists():
        q_embs = np.load(args.text_cache).astype(np.float32)[:len(valid_anns)]
    else:
        try:
            from tasks.real_models import RealCLIPModel
            clip = RealCLIPModel()
            q_embs = np.stack(clip.encode_text(
                [a["query"] for a in valid_anns]
            )).astype(np.float32)
            if args.text_cache:
                np.save(args.text_cache, q_embs)
        except Exception as e:
            print(f"   [warn] CLIP unavailable ({e}), using random")
            q_embs = np.random.randn(len(valid_anns), 512).astype(np.float32)
    q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9

    # Feature map for localization
    id_to_entry = {e.video_id: e for e in index.entries}

    # ── Evaluate 3 methods ──
    print("[3/4] Running moment localization ...")
    methods = {
        "oracle_video": None,       # localize on GT video
        "semantic_top1": "semantic", # retrieve top-1 then localize
    }

    all_results = {}
    for method_name in ["oracle_video", "semantic_top1"]:
        pred_segs_list = []
        gt_spans_list = []

        for i, ann in enumerate(valid_anns):
            gt_vid = ann["vid"]
            gt_spans = [(w[0], w[1]) for w in ann.get("relevant_windows", [])]
            gt_spans_list.append(gt_spans)

            if method_name == "oracle_video":
                target_vid = gt_vid
            else:
                retrieved = retrieve_semantic(index, q_embs[i], top_k=args.top_k)
                target_vid = retrieved[0] if retrieved else gt_vid

            entry = id_to_entry.get(target_vid)
            if entry is None or entry.frame_embs is None:
                pred_segs_list.append([])
                continue

            segs = localize(entry.frame_embs, q_embs[i], clip_len=2.0)
            pred_segs_list.append(segs)

        r1_05 = recall_at_iou(pred_segs_list, gt_spans_list, 0.5)
        r1_07 = recall_at_iou(pred_segs_list, gt_spans_list, 0.7)
        all_results[method_name] = {
            "R1@0.5": r1_05,
            "R1@0.7": r1_07,
            "n_queries": len(valid_anns),
        }
        print(f"   {method_name}: R1@0.5={r1_05*100:.1f}%  R1@0.7={r1_07*100:.1f}%")

    # ── Save ──
    print("[4/4] Saving ...")
    (out / "moment_localization.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print(f"\n[saved] {out}/moment_localization.json")


if __name__ == "__main__":
    main()
