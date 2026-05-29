"""Compute accuracy metrics by comparing a strategy against the 'oracle'.

Since this framework is benchmark-driven with mock models, we define
accuracy by agreement with the 'independent dense' baseline outputs,
which sees every frame at the highest fps. We treat that baseline's
outputs as ground-truth.
"""
from __future__ import annotations

from typing import Dict, List

from core.segment_aggregator import Segment, segments_mean_iou, boundary_mae


def _payload_segs(payload: dict, key: str = "segments") -> List[Segment]:
    """Parse segment dicts from a task payload into Segment objects."""
    raw = payload.get(key, [])
    return [Segment(start=s["start"], end=s["end"], score=s.get("score", 0.0))
            for s in raw if isinstance(s, dict)]


def retrieval_agreement(pred, oracle, tol_sec: float = 2.0) -> float:
    """Agreement rate: fraction of top-1 matches across all queries whose
    predicted top-1 timestamp is within `tol_sec` of the oracle's top-1.
    """
    if not oracle or not pred:
        return 0.0
    p = pred.get("top_k_per_query", [])
    o = oracle.get("top_k_per_query", [])
    if not p or not o:
        return 0.0
    matches = 0
    n = min(len(p), len(o))
    for pi, oi in zip(p[:n], o[:n]):
        if pi and oi:
            dt = abs(pi[0]["timestamp"] - oi[0]["timestamp"])
            if dt <= tol_sec:
                matches += 1
    return matches / n


def highlight_agreement(pred, oracle, iou_thresh: float = 0.3) -> float:
    """Mean IoU of predicted highlight segments vs oracle segments."""
    ps = pred.get("segments", [])
    os_ = oracle.get("segments", [])
    if not ps or not os_:
        return 1.0 if not ps and not os_ else 0.0

    def iou(a, b):
        s = max(a["start"], b["start"])
        e = min(a["end"], b["end"])
        inter = max(0.0, e - s)
        union = max(a["end"], b["end"]) - min(a["start"], b["start"])
        return inter / (union + 1e-9)

    hits = 0
    for a in ps:
        if any(iou(a, b) >= iou_thresh for b in os_):
            hits += 1
    precision = hits / len(ps)
    recall_hits = 0
    for b in os_:
        if any(iou(a, b) >= iou_thresh for a in ps):
            recall_hits += 1
    recall = recall_hits / len(os_)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def face_recall(pred, oracle) -> float:
    pd = {round(d["timestamp"], 1): d["present"]
          for d in pred.get("detections", [])}
    od = {round(d["timestamp"], 1): d["present"]
          for d in oracle.get("detections", [])}
    if not od:
        return 1.0
    pos_oracle = [t for t, v in od.items() if v]
    if not pos_oracle:
        return 1.0
    hits = 0
    for t in pos_oracle:
        # allow +/- 0.5s tolerance
        for delta in (-0.2, -0.1, 0.0, 0.1, 0.2):
            key = round(t + delta, 1)
            if pd.get(key, False):
                hits += 1
                break
    return hits / len(pos_oracle)


def scene_agreement(pred, oracle) -> float:
    """Histogram overlap of scene label distributions.

    More forgiving than exact-match on dominant: if the pred histogram
    covers the oracle's major labels (Jaccard-like), we count it as good.
    """
    p_hist = pred.get("histogram", {}) or {}
    o_hist = oracle.get("histogram", {}) or {}
    if not o_hist:
        return 1.0 if not p_hist else 0.0
    p_total = max(sum(p_hist.values()), 1)
    o_total = max(sum(o_hist.values()), 1)
    p_dist = {k: v / p_total for k, v in p_hist.items()}
    o_dist = {k: v / o_total for k, v in o_hist.items()}
    keys = set(p_dist) | set(o_dist)
    # 1 - total variation distance
    tvd = 0.5 * sum(abs(p_dist.get(k, 0) - o_dist.get(k, 0)) for k in keys)
    return max(0.0, 1.0 - tvd)


def face_emb_coverage(pred, oracle) -> float:
    """Ratio of embeddings produced vs oracle."""
    pn = pred.get("n_embeddings", 0)
    on = oracle.get("n_embeddings", 0)
    if on == 0:
        return 1.0
    return min(1.0, pn / on)


def compute_accuracy_vs_oracle(
    strategy_results: Dict, oracle_results: Dict
) -> Dict[str, float]:
    """Compute one accuracy value per task, vs the oracle output dict.

    Each value is in [0, 1], higher = better.
    Also computes segment-level IoU and boundary MAE where applicable.
    """
    out: Dict[str, float] = {}
    for tid, res in strategy_results.items():
        oracle = oracle_results.get(tid)
        if oracle is None:
            continue
        pred_payload = res.payload or {}
        oracle_payload = oracle.payload or {}
        if tid == "retrieval":
            out[f"{tid}_acc"] = retrieval_agreement(
                pred_payload, oracle_payload
            )
            # Segment IoU across all queries
            pred_segs_all = pred_payload.get("segments_per_query", [])
            orac_segs_all = oracle_payload.get("segments_per_query", [])
            if pred_segs_all and orac_segs_all:
                ious: List[float] = []
                for ps, os_ in zip(pred_segs_all, orac_segs_all):
                    p = [Segment(**s) for s in ps if isinstance(s, dict)]
                    o = [Segment(**s) for s in os_ if isinstance(s, dict)]
                    ious.append(segments_mean_iou(p, o))
                out[f"{tid}_seg_iou"] = sum(ious) / max(len(ious), 1)
        elif tid == "highlight":
            out[f"{tid}_acc"] = highlight_agreement(
                pred_payload, oracle_payload
            )
            p_segs = _payload_segs(pred_payload)
            o_segs = _payload_segs(oracle_payload)
            out[f"{tid}_seg_iou"] = segments_mean_iou(p_segs, o_segs)
            out[f"{tid}_bnd_mae"] = boundary_mae(p_segs, o_segs)
        elif tid == "face_det":
            out[f"{tid}_acc"] = face_recall(pred_payload, oracle_payload)
            p_segs = _payload_segs(pred_payload)
            o_segs = _payload_segs(oracle_payload)
            out[f"{tid}_seg_iou"] = segments_mean_iou(p_segs, o_segs)
            out[f"{tid}_bnd_mae"] = boundary_mae(p_segs, o_segs)
        elif tid == "face_emb":
            out[f"{tid}_acc"] = face_emb_coverage(
                pred_payload, oracle_payload
            )
        elif tid == "scene":
            out[f"{tid}_acc"] = scene_agreement(
                pred_payload, oracle_payload
            )
    return out
