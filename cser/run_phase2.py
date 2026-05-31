"""Phase-2 driver: full CSER pipeline + baselines + experiments E1/E3/E4/E5/E6.

Real-expert version (5 models via mock fallback over a video gallery).

    # synthetic (mock experts, no external data)
    python -m cser.run_phase2 --out-dir reports/cser_phase2

    # real expert models over a real video gallery
    python -m cser.run_phase2 --out-dir reports/cser_phase2 \
        --videos /path/to/videos_dir --csv /path/to/queries.csv --real-models

Outputs under --out-dir:
    e1_main_results.json   CSER vs baselines (R@1/R@5/MRR/cost/coverage)
    e3_conformal.json      coverage vs alpha, split vs Mondrian set sizes
    e4_budget_curve.json   R@1 vs avg experts across budgets
    e5_svn_ablation.json   SVN variants + submod-loss on/off
    e6_safety_ablation.json  conformal vs heuristic vs no-gate
    phase2_summary.json
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
from cser.value_oracle import build_oracle_labels, OracleLabels
from cser.train_svn import train_svn, SVNTrainConfig
from cser.svn import SubmodularValueNetwork
from cser.pipeline import CSERPipeline
from cser.conformal import (SplitConformal, MondrianConformal, gt_nonconformity,
                            qpp_margin, evaluate_coverage)
from cser.baselines import (AllExperts, RandomSelect, FixedCascade, UCBBandit,
                            oracle_mask)
from cser.experts import (N_OPTIONAL, OPTIONAL_COSTS, SEMANTIC_COST,
                          mask_to_names, mask_to_id)
from eval.metrics import retrieval_metrics


class EvalContext:
    """Caches selection-independent quantities for a query split."""

    def __init__(self, engine: RetrievalEngine, oracle: OracleLabels,
                 priors, gt_ids):
        self.engine = engine
        self.oracle = oracle
        self.priors = list(priors)
        self.gt_ids = list(gt_ids)
        self.n = len(self.gt_ids)
        self.sim_norm = [engine.semantic_norm(p) for p in self.priors]
        self.gt_idx = [engine.id_to_idx(g) for g in self.gt_ids]
        self.margins = np.array([qpp_margin(s) for s in self.sim_norm])

    def rank_for_mask(self, i: int, mask: np.ndarray) -> int:
        return self.engine.rank_of_gt(self.priors[i], self.gt_ids[i],
                                      mask_to_names(mask))

    def cost_for_mask(self, mask: np.ndarray) -> float:
        return float(SEMANTIC_COST + OPTIONAL_COSTS[mask].sum())


def _metrics(ranks, costs, ncalls, coverage=None) -> dict:
    out = dict(retrieval_metrics(np.array(ranks, dtype=np.int32)))
    out["avg_cost"] = float(np.mean(costs))
    out["avg_experts_called"] = float(np.mean(ncalls))
    out["GT_filtered_rate"] = 0.0
    if coverage is not None:
        out["conformal_coverage"] = float(np.mean(coverage))
    return out


def _eval_policy(ctx: EvalContext, policy, online_update=False) -> dict:
    ranks, costs, ncalls = [], [], []
    for i in range(ctx.n):
        mask = policy.select(ctx.oracle.query_feats[i])
        r = ctx.rank_for_mask(i, mask)
        ranks.append(r); costs.append(ctx.cost_for_mask(mask))
        ncalls.append(1 + int(mask.sum()))
        if online_update:
            policy.update(0.0 if r < 0 else 1.0 / (r + 1.0))
    return _metrics(ranks, costs, ncalls)


def _eval_oracle(ctx: EvalContext, budget: float) -> dict:
    ranks, costs, ncalls = [], [], []
    for i in range(ctx.n):
        mask = oracle_mask(ctx.oracle.value_matrix[i], budget)
        ranks.append(ctx.rank_for_mask(i, mask))
        costs.append(ctx.cost_for_mask(mask)); ncalls.append(1 + int(mask.sum()))
    return _metrics(ranks, costs, ncalls)


def _eval_cser(ctx: EvalContext, model, gate, budget: float) -> dict:
    pipe = CSERPipeline(ctx.engine, model, conformal_gate=gate, budget=budget)
    ranks, costs, ncalls, cov = [], [], [], []
    for i in range(ctx.n):
        res = pipe.run(ctx.priors[i], ctx.oracle.query_feats[i], ctx.gt_ids[i])
        ranks.append(res.rank); costs.append(res.cost)
        ncalls.append(res.n_experts_called); cov.append(res.gt_in_conformal_set)
    return _metrics(ranks, costs, ncalls, coverage=cov)


# __APPEND_P2_EXPERIMENTS__


# default budget allows ~all experts; B in expert-call units (full set = 9.5)
def exp_e1(ctx, model, gate, budget):
    return {
        "B0_all_experts": _eval_policy(ctx, AllExperts(budget)),
        "B1_random": _eval_policy(ctx, RandomSelect(budget)),
        "B2_fixed_cascade": _eval_policy(ctx, FixedCascade(budget)),
        "B4_ucb_bandit": _eval_policy(ctx, UCBBandit(budget), online_update=True),
        "B_oracle": _eval_oracle(ctx, budget),
        "B6_cser": _eval_cser(ctx, model, gate, budget),
    }


def exp_e3(ctx_cal, ctx_te, alphas=(0.01, 0.05, 0.10, 0.20), n_bins=3):
    cal_scores = np.array([gt_nonconformity(ctx_cal.sim_norm[i], ctx_cal.gt_idx[i])
                           for i in range(ctx_cal.n)])
    out = {}
    for a in alphas:
        split = SplitConformal.calibrate(cal_scores, a)
        mond = MondrianConformal.calibrate(cal_scores, ctx_cal.margins, a, n_bins)
        out[f"alpha={a:.2f}"] = {
            "target_coverage": 1.0 - a,
            "split": evaluate_coverage(split, ctx_te.sim_norm, ctx_te.gt_idx, "split").to_dict(),
            "mondrian": evaluate_coverage(mond, ctx_te.sim_norm, ctx_te.gt_idx, "mondrian").to_dict(),
        }
    return out


def exp_e4(ctx, model, gate, budgets=(1.0, 3.0, 5.0, 7.0, 9.5)):
    out = {}
    for B in budgets:
        out[f"budget={B:.1f}"] = {
            "B0_all_experts": _eval_policy(ctx, AllExperts(B)),
            "B2_fixed_cascade": _eval_policy(ctx, FixedCascade(B)),
            "B_oracle": _eval_oracle(ctx, B),
            "B6_cser": _eval_cser(ctx, model, gate, B),
        }
    return out


def exp_e5(oracle_tr, ctx_te, gate, budget, epochs, seed):
    from cser.run_phase1 import _svn_prediction_submod_violation
    out = {}
    for name, variant, lam in [
        ("full", "full", 0.5),
        ("no_cross_attn", "no_cross_attn", 0.5),
        ("no_set_conditioning", "no_set_conditioning", 0.5),
        ("full_no_submod_loss", "full", 0.0),
    ]:
        cfg = SVNTrainConfig(epochs=epochs, variant=variant, lambda_sub=lam, seed=seed)
        model, _ = train_svn(oracle_tr, cfg, verbose=False)
        m = _eval_cser(ctx_te, model, gate, budget)
        m["svn_pred_submod_violation"] = _svn_prediction_submod_violation(
            model, ctx_te.oracle.query_feats)
        m["param_count"] = model.param_count()
        out[name] = m
    return out


class _HeuristicGate:
    alpha = float("nan")
    def __init__(self, t): self.t = t
    def contains(self, sn, vi): return bool((1.0 - sn[vi]) <= self.t)
    def set_size(self, sn): return int(((1.0 - sn) <= self.t).sum())
    def to_dict(self): return {"kind": "heuristic", "threshold": self.t}


def exp_e6(ctx_cal, ctx_te, model, budget, alpha=0.05):
    cal_scores = np.array([gt_nonconformity(ctx_cal.sim_norm[i], ctx_cal.gt_idx[i])
                           for i in range(ctx_cal.n)])
    split = SplitConformal.calibrate(cal_scores, alpha)
    mond = MondrianConformal.calibrate(cal_scores, ctx_cal.margins, alpha, 3)
    out = {
        "cser_mondrian_conformal": _eval_cser(ctx_te, model, mond, budget),
        "cser_split_conformal": _eval_cser(ctx_te, model, split, budget),
        "cser_heuristic_threshold": _eval_cser(ctx_te, model, _HeuristicGate(0.5), budget),
        "cser_no_gate": _eval_cser(ctx_te, model, None, budget),
    }
    out["cser_mondrian_conformal"]["coverage_report"] = \
        evaluate_coverage(mond, ctx_te.sim_norm, ctx_te.gt_idx, "mondrian").to_dict()
    out["cser_split_conformal"]["coverage_report"] = \
        evaluate_coverage(split, ctx_te.sim_norm, ctx_te.gt_idx, "split").to_dict()
    return out


# __APPEND_P2_MAIN__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/cser_phase2")
    ap.add_argument("--videos", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--real-models", action="store_true")
    ap.add_argument("--metric", default="rr",
                    choices=["rr", "recall@1", "recall@5", "recall@10"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--syn-videos", type=int, default=80)
    ap.add_argument("--syn-queries", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    if args.videos:
        print(f"[data] real video gallery from {args.videos}")
        ds = load_video_dataset(args.videos, args.csv,
                                use_real_models=args.real_models, seed=args.seed)
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
    p, g = _sub(cal_idx)
    oracle_cal = build_oracle_labels(engine, p, g, metric=args.metric, verbose=False)
    p, g = _sub(te_idx)
    oracle_te = build_oracle_labels(engine, p, g, metric=args.metric, verbose=False)

    print("[svn] training production model ...")
    cfg = SVNTrainConfig(epochs=args.epochs, variant="full", seed=args.seed)
    model, _ = train_svn(oracle_tr, cfg, save_dir=str(out / "svn"), verbose=False)

    p_cal, g_cal = _sub(cal_idx)
    ctx_cal = EvalContext(engine, oracle_cal, p_cal, g_cal)
    p_te, g_te = _sub(te_idx)
    ctx_te = EvalContext(engine, oracle_te, p_te, g_te)

    cal_scores = np.array([gt_nonconformity(ctx_cal.sim_norm[i], ctx_cal.gt_idx[i])
                           for i in range(ctx_cal.n)])
    gate = MondrianConformal.calibrate(cal_scores, ctx_cal.margins, args.alpha, 3)

    print("[E1] main comparison ...")
    e1 = exp_e1(ctx_te, model, gate, args.budget)
    (out / "e1_main_results.json").write_text(json.dumps(e1, indent=2, default=str))
    print("[E3] conformal coverage ...")
    e3 = exp_e3(ctx_cal, ctx_te)
    (out / "e3_conformal.json").write_text(json.dumps(e3, indent=2, default=str))
    print("[E4] budget curve ...")
    e4 = exp_e4(ctx_te, model, gate)
    (out / "e4_budget_curve.json").write_text(json.dumps(e4, indent=2, default=str))
    print("[E5] SVN ablation ...")
    e5 = exp_e5(oracle_tr, ctx_te, gate, args.budget, args.epochs, args.seed)
    (out / "e5_svn_ablation.json").write_text(json.dumps(e5, indent=2, default=str))
    print("[E6] safety ablation ...")
    e6 = exp_e6(ctx_cal, ctx_te, model, args.budget, args.alpha)
    (out / "e6_safety_ablation.json").write_text(json.dumps(e6, indent=2, default=str))

    summary = {
        "source": source, "metric": args.metric, "budget": args.budget,
        "alpha": args.alpha, "real_models": args.real_models,
        "gallery_size": ds.gallery_size, "n_queries": ds.n_queries,
        "split": {"train": int(len(tr_idx)), "cal": int(len(cal_idx)),
                  "test": int(len(te_idx))},
        "e1_main_results": e1, "production_gate": gate.to_dict(),
    }
    (out / "phase2_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"E1 main comparison (test n={ctx_te.n}, budget={args.budget}, alpha={args.alpha})")
    print("=" * 78)
    print(f"{'method':<22}{'R@1':>7}{'R@5':>7}{'MRR':>7}{'cost':>7}{'experts':>9}{'cover':>8}")
    print("-" * 78)
    for name, m in e1.items():
        cov = m.get("conformal_coverage", float("nan"))
        cov_s = ("%.1f%%" % (cov * 100)) if cov == cov else "   -"
        print(f"{name:<22}{m['R@1']*100:>6.1f}%{m['R@5']*100:>6.1f}%"
              f"{m['MRR']:>7.3f}{m['avg_cost']:>7.2f}"
              f"{m['avg_experts_called']:>9.2f}{cov_s:>8}")
    print("=" * 78)
    print(f"\n[done] artifacts in {out}/")


if __name__ == "__main__":
    main()

