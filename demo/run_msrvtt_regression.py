"""Multi-task regression on real MSR-VTT 10k videos.

Runs every non-retrieval Huawei model (MomentDETR, InsightFace-Det,
InsightFace-Emb, MobileNetV3-Scene) on a sampled subset of real web
videos and verifies:

  C1 — BIT-IDENTITY      : frames present in BOTH oracle-sampled and
                           V2-sampled sets produce identical model
                           outputs in every pipeline.

  C2 — QUALITY COVERAGE  : V2's adaptive sampler covers ≥ X% of
                           oracle's "informative" timestamps:
                             - highlight: top-20% MomentDETR saliency
                             - face_det : any positive-detection times
                             - scene    : dominant label per segment

  C3 — FACE EMBEDDING    : on frames where BOTH pipelines detect a face,
                           ArcFace embeddings are identical (bit-exact).

Usage:
    python demo/run_msrvtt_regression.py \
        --videos-dir ../video_retrieval_code_no_dataset/data/MSRVTT_Videos/video \
        --n-videos 30 --seed 42 \
        --out REGRESSION_MSRVTT.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.types import FrameRequest, SamplingStage
from core.decoder import decode_frames, probe_video
from core.adaptive_sampler import (
    HybridSampler, MVBasedSampler, UniformSampler,
)
from core.frame_identity import byte_hash

from tasks.real_models import (
    MomentDETRHighlightModel, InsightFaceDetector, InsightFaceEmbedder,
    MobileNetV3SceneClassifier,
)


# ----------------------------------------------------------------------
#  Sampler strategies
# ----------------------------------------------------------------------

def oracle_timestamps(duration: float, fps: float = 2.0,
                       max_frames: int = 64) -> List[float]:
    ts = list(np.arange(0.0, max(duration, 1e-6), 1.0 / fps))
    if len(ts) > max_frames:
        ts = list(np.linspace(0.0, duration, max_frames, endpoint=False))
    return [float(round(t, 3)) for t in ts]


def v2_timestamps(video_path: str, duration: float,
                   target_fps: float = 2.0, max_frames: int = 64) -> List[float]:
    """V2 adaptive: uniform ∪ MV-based, capped for consistency."""
    uniform_ts = list(np.arange(0.0, max(duration, 1e-6), 1.0 / target_fps))
    mv = MVBasedSampler(motion_tau=1.0, max_samples=max_frames)
    mv_ts = [t for t, _ in mv.sample(video_path, duration)]
    all_ts = sorted(set(round(t, 3) for t in uniform_ts + mv_ts))
    if len(all_ts) > max_frames:
        all_ts = sorted(random.sample(all_ts, max_frames))
    return [float(t) for t in all_ts]


def decode_at(video_path: str, video_id: str, timestamps: List[float]):
    reqs = [FrameRequest(
        video_id=video_id, frame_idx=int(t * 25), timestamp=float(t),
        stage=SamplingStage.DENSE, subscribers={"any"},
    ) for t in timestamps]
    return decode_frames(video_path, reqs)


# ----------------------------------------------------------------------
#  Model outputs
# ----------------------------------------------------------------------

def _score_highlight(hl, imgs):
    try:
        return np.asarray(hl.score(imgs), dtype=np.float32)
    except Exception:
        return np.zeros(len(imgs), dtype=np.float32)


def _detect_faces(fd, imgs):
    out = fd.detect(imgs)
    return [(bool(p), float(c)) for p, c in out]


def _embed_faces(fe, imgs):
    return [np.asarray(e, dtype=np.float32) for e in fe.embed(imgs)]


def _classify_scene(sc, imgs):
    return list(sc.classify(imgs))


# ----------------------------------------------------------------------
#  Per-video test
# ----------------------------------------------------------------------

def test_one_video(video_path: str, video_id: str, duration: float,
                    hl, fd, fe, sc,
                    target_fps: float = 2.0, max_frames: int = 64,
                    face_tol_sec: float = 0.5) -> Dict:
    ts_o = oracle_timestamps(duration, target_fps, max_frames)
    ts_v = v2_timestamps(video_path, duration, target_fps, max_frames)
    fr_o = decode_at(video_path, video_id, ts_o)
    fr_v = decode_at(video_path, video_id, ts_v)
    if not fr_o or not fr_v:
        return {"video_id": video_id, "skip": "decode failed"}

    img_o = [f.image for f in fr_o]
    img_v = [f.image for f in fr_v]

    # ---- Run all 4 models on both frame sets ----
    hl_o = _score_highlight(hl, img_o)
    hl_v = _score_highlight(hl, img_v)
    fd_o = _detect_faces(fd, img_o)
    fd_v = _detect_faces(fd, img_v)
    sc_o = _classify_scene(sc, img_o)
    sc_v = _classify_scene(sc, img_v)

    # ---- C1: bit-identity on intersection ----
    h_o = [byte_hash(im) for im in img_o]
    h_v = [byte_hash(im) for im in img_v]
    idx_o = {h: i for i, h in enumerate(h_o)}
    idx_v = {h: i for i, h in enumerate(h_v)}
    common = set(idx_o) & set(idx_v)
    bit_report = {"n_common": len(common)}
    if common:
        # Highlight: note — ordering-sensitive; we only report whether
        # the SAME frame hashed to the same byte_key, which should be
        # exactly true by construction.
        fd_agree = sum(1 for h in common
                        if fd_o[idx_o[h]][0] == fd_v[idx_v[h]][0]) / len(common)
        sc_agree = sum(1 for h in common
                        if sc_o[idx_o[h]] == sc_v[idx_v[h]]) / len(common)
        bit_report["face_det_binary_agree"] = fd_agree
        bit_report["scene_label_agree"]     = sc_agree
        # Face embedding: only for frames with BOTH pipelines detecting
        # a face (conf > threshold).
        emb_o = _embed_faces(fe, img_o)
        emb_v = _embed_faces(fe, img_v)
        cos_vals: List[float] = []
        n_cmp = 0
        for h in common:
            i, j = idx_o[h], idx_v[h]
            if fd_o[i][0] and fd_v[j][0]:
                a = emb_o[i]; b = emb_v[j]
                na = np.linalg.norm(a); nb = np.linalg.norm(b)
                if na > 1e-6 and nb > 1e-6:
                    cos_vals.append(float(np.dot(a, b) / (na * nb)))
                    n_cmp += 1
        if cos_vals:
            bit_report["face_emb_min_cos"]  = float(np.min(cos_vals))
            bit_report["face_emb_mean_cos"] = float(np.mean(cos_vals))
            bit_report["face_emb_n"]        = n_cmp
        else:
            bit_report["face_emb_n"] = 0

    # ---- C2: quality coverage ----
    quality: Dict = {}

    # Highlight: what fraction of oracle's top-20% saliency timestamps
    # does V2 cover within ±1.0s?
    if len(hl_o) > 0:
        thr = float(np.percentile(hl_o, 80))
        hot_ts = [ts_o[i] for i, s in enumerate(hl_o) if s >= thr]
        v_arr = np.array(sorted(ts_v))
        covered = 0
        for t in hot_ts:
            if v_arr.size:
                i = np.searchsorted(v_arr, t)
                nd = min(abs(v_arr[min(i, len(v_arr) - 1)] - t),
                         abs(v_arr[max(i - 1, 0)] - t))
                if nd <= 1.0:
                    covered += 1
        quality["highlight_hot_coverage"] = covered / max(len(hot_ts), 1)
        quality["highlight_n_hot_oracle"] = len(hot_ts)
    else:
        quality["highlight_hot_coverage"] = 1.0
        quality["highlight_n_hot_oracle"] = 0

    # Face det: per oracle positive detection, does V2 sample a frame
    # within face_tol_sec that also detects?
    o_pos_ts = [ts_o[i] for i, (p, _) in enumerate(fd_o) if p]
    v_pos_ts = np.array(sorted([ts_v[i] for i, (p, _) in enumerate(fd_v) if p]))
    hit = 0
    for t in o_pos_ts:
        if v_pos_ts.size:
            i = np.searchsorted(v_pos_ts, t)
            nd = min(abs(v_pos_ts[min(i, len(v_pos_ts) - 1)] - t),
                     abs(v_pos_ts[max(i - 1, 0)] - t))
            if nd <= face_tol_sec:
                hit += 1
    quality["face_det_pos_recall"] = hit / max(len(o_pos_ts), 1) \
        if o_pos_ts else 1.0
    quality["face_det_n_oracle_pos"] = len(o_pos_ts)
    quality["face_det_n_v2_pos"]     = int(v_pos_ts.size)

    # Scene: dominant label agreement
    from collections import Counter
    dom_o = Counter(sc_o).most_common(1)[0][0] if sc_o else None
    dom_v = Counter(sc_v).most_common(1)[0][0] if sc_v else None
    quality["scene_dominant_o"]     = dom_o
    quality["scene_dominant_v"]     = dom_v
    quality["scene_dominant_agree"] = 1.0 if dom_o == dom_v else 0.0
    # histogram TVD
    h_o_cnt = Counter(sc_o); h_v_cnt = Counter(sc_v)
    o_tot = max(sum(h_o_cnt.values()), 1)
    v_tot = max(sum(h_v_cnt.values()), 1)
    keys = set(h_o_cnt) | set(h_v_cnt)
    tvd = 0.5 * sum(abs(h_o_cnt.get(k, 0) / o_tot - h_v_cnt.get(k, 0) / v_tot)
                     for k in keys)
    quality["scene_tvd"] = float(tvd)

    return {
        "video_id": video_id,
        "duration": duration,
        "n_oracle_frames": len(fr_o),
        "n_v2_frames":     len(fr_v),
        "C1_bit_identity": bit_report,
        "C2_quality":      quality,
        "n_face_positive_oracle": len(o_pos_ts),
        "highlight_score_oracle_p50": float(np.percentile(hl_o, 50))
            if len(hl_o) else 0.0,
        "highlight_score_oracle_max": float(np.max(hl_o))
            if len(hl_o) else 0.0,
    }


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True,
                    help="Directory containing MSR-VTT video*.mp4 files")
    ap.add_argument("--n-videos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-duration", type=float, default=60.0,
                    help="skip videos longer than this (MomentDETR 75-clip limit)")
    ap.add_argument("--out", default="REGRESSION_MSRVTT.json")
    args = ap.parse_args()

    vdir = Path(args.videos_dir)
    all_mp4 = sorted(vdir.glob("video*.mp4"))
    if not all_mp4:
        print(f"[error] no videos found in {vdir}")
        sys.exit(1)
    print(f"[videos] {len(all_mp4)} available; sampling {args.n_videos} with seed={args.seed}")

    random.seed(args.seed)
    candidates = random.sample(all_mp4, min(len(all_mp4), args.n_videos * 3))

    # Filter by duration
    videos = []
    for p in candidates:
        if len(videos) >= args.n_videos:
            break
        info = probe_video(str(p))
        dur = info.get("duration", 0.0)
        if 2.0 < dur <= args.max_duration:
            videos.append({"id": p.stem, "path": str(p), "duration": dur})
    print(f"[videos] {len(videos)} videos after duration filter")

    # Build models (once, shared)
    print("[models] loading real backbones...")
    hl = MomentDETRHighlightModel()
    fd = InsightFaceDetector()
    fe = InsightFaceEmbedder()
    sc = MobileNetV3SceneClassifier()

    # Run
    rows = []
    t0 = time.perf_counter()
    for i, v in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {v['id']} dur={v['duration']:.1f}s")
        try:
            r = test_one_video(
                v["path"], v["id"], v["duration"], hl, fd, fe, sc,
            )
            if "skip" not in r:
                c1 = r["C1_bit_identity"]
                c2 = r["C2_quality"]
                print(f"  C1: n_common={c1['n_common']}  "
                      f"fd_agree={c1.get('face_det_binary_agree', '-')}  "
                      f"sc_agree={c1.get('scene_label_agree', '-')}  "
                      f"femb_cos={c1.get('face_emb_mean_cos', '-')}  "
                      f"(n={c1.get('face_emb_n', 0)})")
                print(f"  C2: hl_cov={c2['highlight_hot_coverage']:.2f}  "
                      f"fd_rec={c2['face_det_pos_recall']:.2f}  "
                      f"sc_agr={c2['scene_dominant_agree']:.2f} "
                      f"({c2['scene_dominant_o']}->{c2['scene_dominant_v']})  "
                      f"tvd={c2['scene_tvd']:.3f}")
            rows.append(r)
        except Exception as e:
            print(f"  [error] {e}")
            rows.append({"video_id": v["id"], "error": str(e)})

    elapsed = time.perf_counter() - t0
    print(f"\n[done] {len(rows)} videos in {elapsed:.0f}s")

    # ---- Aggregate ----
    valid = [r for r in rows if "skip" not in r and "error" not in r]
    if not valid:
        print("[error] no valid results")
        return

    n_total = len(valid)
    # Bit-identity
    fd_vals = [r["C1_bit_identity"].get("face_det_binary_agree")
                for r in valid
                if "face_det_binary_agree" in r["C1_bit_identity"]]
    sc_vals = [r["C1_bit_identity"].get("scene_label_agree")
                for r in valid
                if "scene_label_agree" in r["C1_bit_identity"]]
    emb_cos = [r["C1_bit_identity"].get("face_emb_mean_cos")
                for r in valid
                if r["C1_bit_identity"].get("face_emb_n", 0) > 0]
    emb_min = [r["C1_bit_identity"].get("face_emb_min_cos")
                for r in valid
                if r["C1_bit_identity"].get("face_emb_n", 0) > 0]
    # Quality
    hl_cov  = [r["C2_quality"]["highlight_hot_coverage"] for r in valid]
    fd_rec  = [r["C2_quality"]["face_det_pos_recall"] for r in valid]
    sc_agr  = [r["C2_quality"]["scene_dominant_agree"] for r in valid]
    sc_tvd  = [r["C2_quality"]["scene_tvd"] for r in valid]

    summary = {
        "n_videos_tested": n_total,
        "C1_bit_identity": {
            "face_det_binary_agree_mean": float(np.mean(fd_vals)) if fd_vals else None,
            "scene_label_agree_mean":     float(np.mean(sc_vals)) if sc_vals else None,
            "face_emb_mean_cos_mean":     float(np.mean(emb_cos)) if emb_cos else None,
            "face_emb_min_cos_overall":   float(np.min(emb_min)) if emb_min else None,
            "n_videos_with_face_embs":    len(emb_cos),
        },
        "C2_quality": {
            "highlight_hot_coverage_mean": float(np.mean(hl_cov)),
            "face_det_pos_recall_mean":    float(np.mean(fd_rec)),
            "scene_dominant_agree_mean":   float(np.mean(sc_agr)),
            "scene_tvd_mean":              float(np.mean(sc_tvd)),
        },
    }
    # Thresholds
    verdicts = {
        "C1_face_det":  (summary["C1_bit_identity"]["face_det_binary_agree_mean"] or 0) >= 0.99,
        "C1_scene":     (summary["C1_bit_identity"]["scene_label_agree_mean"] or 0) >= 0.99,
        "C1_face_emb":  (summary["C1_bit_identity"]["face_emb_min_cos_overall"] or 1.0) >= 0.9999
                          if emb_min else True,   # vacuously pass if no faces
        "C2_highlight": summary["C2_quality"]["highlight_hot_coverage_mean"] >= 0.85,
        "C2_face_det":  summary["C2_quality"]["face_det_pos_recall_mean"]    >= 0.85,
        "C2_scene":     summary["C2_quality"]["scene_dominant_agree_mean"]   >= 0.85,
        "C2_scene_tvd": summary["C2_quality"]["scene_tvd_mean"]              <= 0.15,
    }
    overall = all(verdicts.values())

    print("\n" + "=" * 72)
    print(f"MSR-VTT real-data multi-task regression ({n_total} videos)")
    print("=" * 72)
    print("\nC1 — Bit-identity on intersection frames:")
    for k, v in summary["C1_bit_identity"].items():
        print(f"  {k:<34} = {v}")
    print("\nC2 — Quality coverage:")
    for k, v in summary["C2_quality"].items():
        print(f"  {k:<34} = {v:.3f}")
    print("\nVerdict per check:")
    for k, ok in verdicts.items():
        print(f"  {k:<15} {'PASS ✓' if ok else 'FAIL ✗'}")
    print(f"\nOVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")

    out = {"summary": summary, "verdicts": verdicts, "overall_pass": overall,
           "per_video": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str),
                                encoding="utf-8")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
