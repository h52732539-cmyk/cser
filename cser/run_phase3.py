"""Phase-3 driver: experiments E7-E10 + empirical theorem verification.

Uses mock experts for synthetic runs or all 5 real experts when ``--real-models``
is requested. Explicit real mode fails closed if any backbone cannot initialize.

    # synthetic (mock experts)
    python -m cser.run_phase3 --out-dir reports/cser_phase3

    # real expert models over a real video gallery
    python -m cser.run_phase3 --out-dir reports/cser_phase3 \
        --videos /path/to/videos_dir --csv /path/to/queries.csv --real-models

Outputs under --out-dir:
    e7_scalability.json / e8_robustness.json / e9_expert_contribution.json
    e10_oracle_comparison.json / theorem_verification.json / phase3_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cser.data import build_synthetic_dataset, load_video_dataset
from cser.retrieval import RetrievalEngine
from cser.value_oracle import build_oracle_labels
from cser.train_svn import train_svn, SVNTrainConfig
from cser.conformal import MondrianConformal, gt_nonconformity, qpp_margin
from cser.submodularity import verify_submodularity
from cser.experts import (N_OPTIONAL, OPTIONAL_COSTS, SEMANTIC_COST,
                          all_optional_masks)
from cser.experiments_extra import (exp_e7_scalability, exp_e8_robustness,
                                    exp_e9_expert_contribution,
                                    exp_e10_oracle_comparison)
from cser.theory import (verify_theorem1_coverage, verify_theorem2_greedy,
                         verify_theorem3_combined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/cser_phase3")
    ap.add_argument("--videos", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--real-models", action="store_true")
    ap.add_argument("--gallery-cache", default=None,
                    help="directory for reusable gallery expert cache")
    ap.add_argument("--metric", default="rr",
                    choices=["rr", "recall@1", "recall@5", "recall@10"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--train-device", default="auto",
                    help="SVN training device: auto, cpu, cuda, or cuda:N")
    ap.add_argument("--train-batch-size", type=int, default=256)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--candidate-top-k", type=int, default=100,
                    help="semantic candidates retained before safety-gate protection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--syn-videos", type=int, default=500)
    ap.add_argument("--syn-queries", type=int, default=300)
    args = ap.parse_args()
    if args.candidate_top_k <= 0:
        ap.error("--candidate-top-k must be positive")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    if args.videos:
        print(f"[data] real video gallery from {args.videos}")
        ds = load_video_dataset(args.videos, args.csv,
                                use_real_models=args.real_models,
                                cache_dir=args.gallery_cache,
                                seed=args.seed)
        source = "video"
    else:
        print("[data] synthetic gallery (mock experts)")
        ds = build_synthetic_dataset(n_videos=args.syn_videos,
                                     n_queries=args.syn_queries,
                                     use_real_models=args.real_models, seed=args.seed)
        source = "synthetic"
    print(f"       source={source} gallery={ds.gallery_size} queries={ds.n_queries}")

    tr_idx, cal_idx, te_idx = ds.split(seed=args.seed)
    engine = RetrievalEngine(ds.gallery)

    def _sub(idx):
        return ([ds.query_priors[i] for i in idx], [ds.gt_video_ids[i] for i in idx])

    print("[oracle] building lattices (train/cal/test) ...")
    p, g = _sub(tr_idx)
    oracle_tr = build_oracle_labels(engine, p, g, metric=args.metric, verbose=False)
    p_cal, g_cal = _sub(cal_idx)
    oracle_cal = build_oracle_labels(engine, p_cal, g_cal, metric=args.metric, verbose=False)
    p_te, g_te = _sub(te_idx)
    oracle_te = build_oracle_labels(engine, p_te, g_te, metric=args.metric, verbose=False)

    print("[svn] training ...")
    cfg = SVNTrainConfig(
        epochs=args.epochs,
        variant="full",
        seed=args.seed,
        device=args.train_device,
        batch_size=args.train_batch_size,
    )
    model, _ = train_svn(oracle_tr, cfg, save_dir=str(out / "svn"), verbose=False)

    # ── Conformal gate (Theorem 1) ──
    cal_sim = [engine.semantic_norm(p) for p in p_cal]
    cal_gidx = [engine.id_to_idx(g) for g in g_cal]
    cal_margins = np.array([qpp_margin(s) for s in cal_sim])
    cal_scores = np.array([gt_nonconformity(cal_sim[k], cal_gidx[k])
                           for k in range(len(g_cal))])
    gate = MondrianConformal.calibrate(cal_scores, cal_margins, args.alpha, 3)
    te_sim = [engine.semantic_norm(p) for p in p_te]
    te_gidx = [engine.id_to_idx(g) for g in g_te]

    print("[E7] scalability ...")
    e7 = exp_e7_scalability(ds, model, budget=args.budget, seed=args.seed)
    (out / "e7_scalability.json").write_text(json.dumps(e7, indent=2, default=str))
    print("[E8] robustness ...")
    e8 = exp_e8_robustness(ds, model, budget=args.budget, seed=args.seed,
                           alpha=args.alpha, candidate_top_k=args.candidate_top_k)
    (out / "e8_robustness.json").write_text(json.dumps(e8, indent=2, default=str))
    print("[E9] expert contribution ...")
    e9 = exp_e9_expert_contribution(oracle_te, model)
    (out / "e9_expert_contribution.json").write_text(json.dumps(e9, indent=2, default=str))
    print("[E10] oracle comparison ...")
    e10 = exp_e10_oracle_comparison(oracle_te, model, budget=args.budget)
    (out / "e10_oracle_comparison.json").write_text(json.dumps(e10, indent=2, default=str))

    print("[theory] verifying theorem bounds ...")
    submod = verify_submodularity(oracle_te)
    thm1 = verify_theorem1_coverage(gate, te_sim, te_gidx)
    thm2 = verify_theorem2_greedy(model, oracle_te, submod.gamma_ratio_p10,
                                  budget=args.budget,
                                  monotonicity_violation_rate=(
                                      submod.monotonicity_violation_rate))
    feasible_max = max(
        SEMANTIC_COST + OPTIONAL_COSTS[m].sum()
        for m in all_optional_masks()
        if SEMANTIC_COST + OPTIONAL_COSTS[m].sum() <= args.budget + 1e-9)
    thm3 = verify_theorem3_combined(thm1, thm2, feasible_max, args.budget)
    theorems = {"submodularity": submod.to_dict(), "theorem1_coverage": thm1,
                "theorem2_greedy": thm2, "theorem3_combined": thm3}
    (out / "theorem_verification.json").write_text(json.dumps(theorems, indent=2, default=str))

    summary = {
        "source": source, "metric": args.metric, "budget": args.budget,
        "alpha": args.alpha, "real_models": args.real_models,
        "candidate_top_k": args.candidate_top_k,
        "latency_scope": "cached_score_rerank_only",
        "gallery_size": ds.gallery_size,
        "n_videos_total": int(ds.n_videos_total),
        "n_videos_loaded": int(ds.gallery_size),
        "failed_video_ids": list(ds.failed_video_ids),
        "gallery_cache_manifest": ds.cache_manifest,
        "n_queries": ds.n_queries,
        "e7_scalability": e7, "e8_robustness": e8,
        "e9_expert_contribution": e9, "e10_oracle_comparison": e10,
        "theorem_verification": theorems,
    }
    (out / "phase3_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 70)
    print("PHASE-3 SUMMARY")
    print("=" * 70)
    print(f"Theorem 1 (coverage):  empirical={thm1['empirical_coverage']:.3f} "
          f"target={thm1['target_coverage']:.3f}  holds={thm1['holds']}")
    print(f"Theorem 2 (greedy):    LHS={thm2['realised_value_LHS']:.3f} "
          f"RHS={thm2['bound_RHS']:.3f}  eps={thm2['surrogate_error_eps']:.4f}  "
          f"holds={thm2['bound_holds']}  non_vacuous={thm2['bound_is_non_vacuous']}")
    print(f"Theorem 3 (combined):  all_hold={thm3['all_three_hold']}")
    print(f"E10 CSER % of oracle:  {e10['cser_svn_greedy']['pct_of_oracle']:.1%}")
    print(f"E9 expert ranking:     {e9['expert_ranking_by_value']}")
    print(f"E9 SVN-oracle corr:    {e9['svn_oracle_marginal_correlation']:.3f}")
    print("=" * 70)
    print(f"\n[done] artifacts in {out}/")


if __name__ == "__main__":
    main()
