"""Phase-2 driver: full CSER pipeline + baselines + experiments E1/E3/E4/E5/E6.

Uses mock experts for synthetic runs or all 5 real experts when ``--real-models``
is requested. Explicit real mode fails closed if any backbone cannot initialize.

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
from cser.train_set_value import train_set_value, SetValueTrainConfig
from cser.pipeline import CSERPipeline
from cser.conformal import (SplitConformal, MondrianConformal, gt_nonconformity,
                            qpp_margin, evaluate_coverage)
from cser.baselines import (AllExperts, RandomSelect, FixedCascade, UCBBandit,
                            oracle_mask)
from cser.selectors import (SELECTOR_MODES, build_selector, load_set_value_model,
                            roster_allowed_mask)
from cser.experts import (N_OPTIONAL, OPTIONAL_COSTS, SEMANTIC_COST,
                          OPTIONAL_NAMES, mask_to_names, mask_to_id)
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


def _rr(rank: int) -> float:
    return 0.0 if rank < 0 else 1.0 / (rank + 1.0)


def _metrics(ranks, costs, ncalls, coverage=None, gt_filtered=None,
             candidate_counts=None, gallery_size=None,
             fallback_triggered=None, safety_mode=None) -> dict:
    out = dict(retrieval_metrics(np.array(ranks, dtype=np.int32)))
    out["avg_cost"] = float(np.mean(costs))
    out["cost_kind"] = "offline_index_expert_unit_proxy"
    out["avg_experts_called"] = float(np.mean(ncalls))
    if safety_mode is not None:
        out["safety_mode"] = safety_mode
    filtered = np.zeros(len(ranks), dtype=bool) if gt_filtered is None \
        else np.asarray(gt_filtered, dtype=bool)
    out["GT_filtered_rate"] = float(filtered.mean())
    if candidate_counts is not None:
        counts = np.asarray(candidate_counts, dtype=np.int32)
        out["avg_candidates_after_filter"] = float(counts.mean())
        out["candidate_count_p50"] = float(np.percentile(counts, 50))
        out["candidate_count_p90"] = float(np.percentile(counts, 90))
        out["candidate_count_p95"] = float(np.percentile(counts, 95))
        out["candidate_count_p99"] = float(np.percentile(counts, 99))
        if gallery_size is not None and gallery_size > 0:
            out["candidate_reduction_rate"] = float(1.0 - counts.mean() / gallery_size)
            out["hard_filter_activation_rate"] = float((counts < gallery_size).mean())
    if coverage is not None:
        out["conformal_coverage"] = float(np.mean(coverage))
    if fallback_triggered is not None:
        out["fallback_triggered_rate"] = float(np.mean(fallback_triggered))
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


def _run_cser(ctx: EvalContext, model, gate, budget: float,
              candidate_top_k: int = 100, selector=None,
              safety_mode: str = "reduce"):
    pipe = CSERPipeline(ctx.engine, model, conformal_gate=gate, budget=budget,
                        candidate_top_k=candidate_top_k, selector=selector,
                        safety_mode=safety_mode)
    return [
        pipe.run(ctx.priors[i], ctx.oracle.query_feats[i], ctx.gt_ids[i])
        for i in range(ctx.n)
    ]


def _eval_cser(ctx: EvalContext, model, gate, budget: float,
               candidate_top_k: int = 100, selector=None,
               safety_mode: str = "reduce") -> dict:
    results = _run_cser(ctx, model, gate, budget, candidate_top_k,
                        selector=selector, safety_mode=safety_mode)
    ranks, costs, ncalls, cov, filtered, candidate_counts = [], [], [], [], [], []
    fallbacks = []
    for res in results:
        ranks.append(res.rank); costs.append(res.cost)
        ncalls.append(res.n_experts_called); filtered.append(res.gt_filtered)
        candidate_counts.append(res.candidate_count)
        fallbacks.append(bool(getattr(res, "fallback_triggered", False)))
        if gate is not None:
            cov.append(res.gt_in_conformal_set)
    return _metrics(ranks, costs, ncalls, coverage=cov or None,
                    gt_filtered=filtered, candidate_counts=candidate_counts,
                    gallery_size=ctx.engine._N,
                    fallback_triggered=(
                        fallbacks if (any(fallbacks) or getattr(selector, "safe", False))
                        else None),
                    safety_mode=safety_mode)


# __APPEND_P2_EXPERIMENTS__


# default budget allows ~all experts; B in expert-call units (full set = 9.5)
def exp_e1(ctx, model, gate, budget, candidate_top_k=100,
           selector=None, safety_mode="reduce"):
    return {
        "B0_all_experts": _eval_policy(ctx, AllExperts(budget)),
        "B1_random": _eval_policy(ctx, RandomSelect(budget)),
        "B2_fixed_cascade": _eval_policy(ctx, FixedCascade(budget)),
        "B4_ucb_bandit": _eval_policy(ctx, UCBBandit(budget), online_update=True),
        "B_oracle": _eval_oracle(ctx, budget),
        "B6_cser": _eval_cser(ctx, model, gate, budget, candidate_top_k,
                              selector=selector, safety_mode=safety_mode),
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


def exp_e4(ctx, model, gate, budgets=(1.0, 3.0, 5.0, 7.0, 9.5),
           candidate_top_k=100):
    out = {}
    for B in budgets:
        out[f"budget={B:.1f}"] = {
            "B0_all_experts": _eval_policy(ctx, AllExperts(B)),
            "B2_fixed_cascade": _eval_policy(ctx, FixedCascade(B)),
            "B_oracle": _eval_oracle(ctx, B),
            "B6_cser": _eval_cser(ctx, model, gate, B, candidate_top_k),
        }
    return out


def exp_e5(oracle_tr, ctx_te, gate, budget, epochs, seed, candidate_top_k=100,
           train_device="auto", train_batch_size=256):
    from cser.run_phase1 import _svn_prediction_submod_violation
    out = {}
    for name, variant, lam in [
        ("full", "full", 0.5),
        ("no_cross_attn", "no_cross_attn", 0.5),
        ("no_set_conditioning", "no_set_conditioning", 0.5),
        ("full_no_submod_loss", "full", 0.0),
    ]:
        cfg = SVNTrainConfig(
            epochs=epochs,
            variant=variant,
            lambda_sub=lam,
            seed=seed,
            device=train_device,
            batch_size=train_batch_size,
        )
        model, _ = train_svn(oracle_tr, cfg, verbose=False)
        m = _eval_cser(ctx_te, model, gate, budget, candidate_top_k)
        m["svn_pred_submod_violation"] = _svn_prediction_submod_violation(
            model, ctx_te.oracle.query_feats)
        m["param_count"] = model.param_count()
        out[name] = m
    return out


class _HeuristicGate:
    alpha = float("nan")
    def __init__(self, t): self.t = t
    def contains(self, sn, vi): return bool((1.0 - sn[vi]) <= self.t)
    def prediction_set_mask(self, sn): return (1.0 - sn) <= self.t
    def set_size(self, sn): return int(self.prediction_set_mask(sn).sum())
    def to_dict(self): return {"kind": "heuristic", "threshold": self.t}


def exp_e6(ctx_cal, ctx_te, model, budget, alpha=0.05, candidate_top_k=100,
           selector=None):
    cal_scores = np.array([gt_nonconformity(ctx_cal.sim_norm[i], ctx_cal.gt_idx[i])
                           for i in range(ctx_cal.n)])
    split = SplitConformal.calibrate(cal_scores, alpha)
    mond = MondrianConformal.calibrate(cal_scores, ctx_cal.margins, alpha, 3)
    out = {
        "Mondrian_reduce": _eval_cser(
            ctx_te, model, mond, budget, candidate_top_k, selector=selector,
            safety_mode="reduce"),
        "Split_reduce": _eval_cser(
            ctx_te, model, split, budget, candidate_top_k, selector=selector,
            safety_mode="reduce"),
        "heuristic_reduce": _eval_cser(
            ctx_te, model, _HeuristicGate(0.5), budget, candidate_top_k,
            selector=selector, safety_mode="reduce"),
        "no_gate": _eval_cser(
            ctx_te, model, None, budget, candidate_top_k, selector=selector,
            safety_mode="reduce"),
        "Mondrian_report": _eval_cser(
            ctx_te, model, mond, budget, candidate_top_k, selector=selector,
            safety_mode="report"),
        "Split_report": _eval_cser(
            ctx_te, model, split, budget, candidate_top_k, selector=selector,
            safety_mode="report"),
        "no_filter": _eval_cser(
            ctx_te, model, None, budget, None, selector=selector,
            safety_mode="reduce"),
    }
    out["Mondrian_reduce"]["coverage_report"] = \
        evaluate_coverage(mond, ctx_te.sim_norm, ctx_te.gt_idx, "mondrian").to_dict()
    out["Split_reduce"]["coverage_report"] = \
        evaluate_coverage(split, ctx_te.sim_norm, ctx_te.gt_idx, "split").to_dict()
    out["Mondrian_report"]["coverage_report"] = \
        evaluate_coverage(mond, ctx_te.sim_norm, ctx_te.gt_idx, "mondrian").to_dict()
    out["Split_report"]["coverage_report"] = \
        evaluate_coverage(split, ctx_te.sim_norm, ctx_te.gt_idx, "split").to_dict()
    return out


SAFETY_CONFIGS = (
    "no_gate",
    "no_filter",
    "Mondrian_reduce",
    "Split_reduce",
    "heuristic_reduce",
    "Mondrian_report",
    "Split_report",
)


def _parse_int_list(raw: str) -> list[int]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("list must contain at least one integer")
    return vals


def _safety_config(config: str, split, mond, heuristic, candidate_top_k: int):
    if config == "no_gate":
        return None, "reduce", candidate_top_k
    if config == "no_filter":
        return None, "reduce", None
    if config == "Mondrian_reduce":
        return mond, "reduce", candidate_top_k
    if config == "Split_reduce":
        return split, "reduce", candidate_top_k
    if config == "heuristic_reduce":
        return heuristic, "reduce", candidate_top_k
    if config == "Mondrian_report":
        return mond, "report", candidate_top_k
    if config == "Split_report":
        return split, "report", candidate_top_k
    raise ValueError(f"unknown safety config: {config}")


def _write_query_audit(path: Path, ctx: EvalContext, results, budget: float) -> None:
    empty = np.zeros(N_OPTIONAL, dtype=bool)
    lines = []
    for i, res in enumerate(results):
        selected = np.asarray(res.selected_mask, dtype=bool)
        oracle = oracle_mask(ctx.oracle.value_matrix[i], budget)
        sem_rank = ctx.rank_for_mask(i, empty)
        cser_rank = int(res.rank)
        oracle_rank = ctx.rank_for_mask(i, oracle)
        sem_rr = _rr(sem_rank)
        cser_rr = _rr(cser_rank)
        oracle_rr = _rr(oracle_rank)
        rec = {
            "query_index": int(i),
            "gt_video_id": ctx.gt_ids[i],
            "selected_mask_id": int(mask_to_id(selected)),
            "selected_experts": mask_to_names(selected),
            "semantic_only_rank": int(sem_rank),
            "semantic_only_rr": float(sem_rr),
            "cser_rank": int(cser_rank),
            "cser_rr": float(cser_rr),
            "oracle_mask_id": int(mask_to_id(oracle)),
            "oracle_experts": mask_to_names(oracle),
            "oracle_rank": int(oracle_rank),
            "oracle_rr": float(oracle_rr),
            "delta_rr_vs_semantic": float(cser_rr - sem_rr),
            "oracle_gap_rr": float(oracle_rr - cser_rr),
            "candidate_count": int(res.candidate_count),
            "gt_filtered": bool(res.gt_filtered),
            "fallback_triggered": bool(getattr(res, "fallback_triggered", False)),
        }
        lines.append(json.dumps(rec, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_mask_distribution(path: Path, ctx: EvalContext, results) -> dict:
    empty = np.zeros(N_OPTIONAL, dtype=bool)
    semantic_rr = np.array([_rr(ctx.rank_for_mask(i, empty)) for i in range(ctx.n)])
    by_mask = {}
    expert_counts = np.zeros(N_OPTIONAL, dtype=np.float64)
    fallbacks = []
    for i, res in enumerate(results):
        mask = np.asarray(res.selected_mask, dtype=bool)
        sid = int(mask_to_id(mask))
        rr = _rr(res.rank)
        rec = by_mask.setdefault(str(sid), {
            "count": 0, "rr_sum": 0.0, "r1_sum": 0.0, "delta_rr_sum": 0.0,
            "experts": mask_to_names(mask),
        })
        rec["count"] += 1
        rec["rr_sum"] += rr
        rec["r1_sum"] += float(res.rank == 0)
        rec["delta_rr_sum"] += rr - float(semantic_rr[i])
        expert_counts += mask.astype(np.float64)
        fallbacks.append(bool(getattr(res, "fallback_triggered", False)))

    masks_out = {}
    for sid, rec in by_mask.items():
        n = max(rec["count"], 1)
        masks_out[sid] = {
            "count": int(rec["count"]),
            "frequency": float(rec["count"] / max(ctx.n, 1)),
            "experts": rec["experts"],
            "avg_rr": float(rec["rr_sum"] / n),
            "avg_R@1": float(rec["r1_sum"] / n),
            "avg_delta_rr_vs_semantic": float(rec["delta_rr_sum"] / n),
        }
    out = {
        "n_queries": int(ctx.n),
        "mask_frequencies": masks_out,
        "expert_selection_frequency": {
            OPTIONAL_NAMES[j]: float(expert_counts[j] / max(ctx.n, 1))
            for j in range(N_OPTIONAL)
        },
        "fallback_triggered_rate": float(np.mean(fallbacks)) if fallbacks else 0.0,
    }
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _write_expert_delta_summary(path: Path, ctx: EvalContext, results,
                                budget: float) -> dict:
    empty = np.zeros(N_OPTIONAL, dtype=bool)
    semantic_rr = np.array([_rr(ctx.rank_for_mask(i, empty)) for i in range(ctx.n)])
    selected_masks = [np.asarray(r.selected_mask, dtype=bool) for r in results]
    oracle_masks = [oracle_mask(ctx.oracle.value_matrix[i], budget)
                    for i in range(ctx.n)]
    out = {}
    for j, name in enumerate(OPTIONAL_NAMES):
        solo = np.zeros(N_OPTIONAL, dtype=bool)
        solo[j] = True
        solo_rr = np.array([_rr(ctx.rank_for_mask(i, solo)) for i in range(ctx.n)])
        selected = np.array([m[j] for m in selected_masks], dtype=bool)
        oracle_sel = np.array([m[j] for m in oracle_masks], dtype=bool)
        cser_rr = np.array([_rr(r.rank) for r in results])
        oracle_rr = np.array([
            _rr(ctx.rank_for_mask(i, oracle_masks[i])) for i in range(ctx.n)
        ])
        out[name] = {
            "solo_add_delta_rr_mean": float((solo_rr - semantic_rr).mean()),
            "selector_selected_rate": float(selected.mean()),
            "selector_selected_delta_rr_mean": (
                float((cser_rr[selected] - semantic_rr[selected]).mean())
                if selected.any() else 0.0
            ),
            "oracle_selected_rate": float(oracle_sel.mean()),
            "oracle_selected_delta_rr_mean": (
                float((oracle_rr[oracle_sel] - semantic_rr[oracle_sel]).mean())
                if oracle_sel.any() else 0.0
            ),
        }
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _candidate_top_k_sweep(ctx_te: EvalContext, model, selector, budget: float,
                           split, mond, heuristic, top_ks: list[int]) -> dict:
    out = {}
    for top_k in top_ks:
        out[str(top_k)] = {
            "no_gate": _eval_cser(ctx_te, model, None, budget, top_k,
                                  selector=selector, safety_mode="reduce"),
            "Mondrian_reduce": _eval_cser(ctx_te, model, mond, budget, top_k,
                                          selector=selector, safety_mode="reduce"),
            "Split_reduce": _eval_cser(ctx_te, model, split, budget, top_k,
                                       selector=selector, safety_mode="reduce"),
            "heuristic_reduce": _eval_cser(ctx_te, model, heuristic, budget, top_k,
                                           selector=selector, safety_mode="reduce"),
        }
    return out


# __APPEND_P2_MAIN__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/cser_phase2")
    ap.add_argument("--videos", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--real-models", action="store_true")
    ap.add_argument("--gallery-cache", default=None,
                    help="directory for reusable gallery expert cache")
    ap.add_argument("--metric", default="rr",
                    choices=["rr", "recall@1", "recall@5", "recall@10"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--train-device", default="auto",
                    help="training device: auto, cpu, cuda, or cuda:N")
    ap.add_argument("--train-batch-size", type=int, default=256)
    ap.add_argument("--set-value-batch-size", type=int, default=128)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--candidate-top-k", type=int, default=100,
                    help="semantic candidates retained before safety-gate protection")
    ap.add_argument("--candidate-top-k-list", default=None,
                    help="comma-separated top-k sweep, e.g. 100,300,500,1000")
    ap.add_argument("--selector", default="marginal_value_greedy",
                    choices=SELECTOR_MODES)
    ap.add_argument("--selector-model", default=None,
                    help="path to a trained set_value.pt for set-value selectors")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="set_value_safe fallback margin over semantic-only prediction")
    ap.add_argument("--expert-roster", default="all",
                    help="all, no_face_id, semantic_highlight_scene, or comma list")
    ap.add_argument("--safety-config", default="Mondrian_reduce",
                    choices=SAFETY_CONFIGS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--syn-videos", type=int, default=80)
    ap.add_argument("--syn-queries", type=int, default=200)
    args = ap.parse_args()
    if args.candidate_top_k <= 0:
        ap.error("--candidate-top-k must be positive")
    try:
        roster_allowed_mask(args.expert_roster)
    except ValueError as exc:
        ap.error(str(exc))
    candidate_top_k_list = None
    if args.candidate_top_k_list:
        try:
            candidate_top_k_list = _parse_int_list(args.candidate_top_k_list)
        except ValueError as exc:
            ap.error(str(exc))
        if any(k <= 0 for k in candidate_top_k_list):
            ap.error("--candidate-top-k-list values must be positive")

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
    p, g = _sub(cal_idx)
    oracle_cal = build_oracle_labels(engine, p, g, metric=args.metric, verbose=False)
    p, g = _sub(te_idx)
    oracle_te = build_oracle_labels(engine, p, g, metric=args.metric, verbose=False)

    print("[svn] training production model ...")
    cfg = SVNTrainConfig(
        epochs=args.epochs,
        variant="full",
        seed=args.seed,
        device=args.train_device,
        batch_size=args.train_batch_size,
    )
    model, _ = train_svn(oracle_tr, cfg, save_dir=str(out / "svn"), verbose=False)

    set_value_model = None
    if args.selector in ("set_value", "set_value_safe"):
        if args.selector_model:
            set_value_model = load_set_value_model(
                args.selector_model, d_query=oracle_tr.feature_dim)
            print(f"[set-value] loaded selector model {args.selector_model}")
        else:
            print("[set-value] training production model ...")
            sv_cfg = SetValueTrainConfig(
                epochs=args.epochs,
                seed=args.seed,
                device=args.train_device,
                batch_size=args.set_value_batch_size,
            )
            set_value_model, _ = train_set_value(
                oracle_tr, sv_cfg, save_dir=str(out / "set_value"), verbose=False)

    p_cal, g_cal = _sub(cal_idx)
    ctx_cal = EvalContext(engine, oracle_cal, p_cal, g_cal)
    p_te, g_te = _sub(te_idx)
    ctx_te = EvalContext(engine, oracle_te, p_te, g_te)

    cal_scores = np.array([gt_nonconformity(ctx_cal.sim_norm[i], ctx_cal.gt_idx[i])
                           for i in range(ctx_cal.n)])
    split_gate = SplitConformal.calibrate(cal_scores, args.alpha)
    mond_gate = MondrianConformal.calibrate(cal_scores, ctx_cal.margins, args.alpha, 3)
    heuristic_gate = _HeuristicGate(0.5)
    gate, safety_mode, e1_candidate_top_k = _safety_config(
        args.safety_config, split_gate, mond_gate, heuristic_gate,
        args.candidate_top_k)

    selector = build_selector(
        args.selector,
        budget=args.budget,
        roster=args.expert_roster,
        svn_model=model,
        set_value_model=set_value_model,
        min_delta=args.min_delta,
    )

    print("[E1] main comparison ...")
    e1_results = _run_cser(ctx_te, model, gate, args.budget, e1_candidate_top_k,
                           selector=selector, safety_mode=safety_mode)
    e1 = exp_e1(ctx_te, model, gate, args.budget, e1_candidate_top_k,
                selector=selector, safety_mode=safety_mode)
    (out / "e1_main_results.json").write_text(json.dumps(e1, indent=2, default=str))
    print("[diagnostics] writing selector audit files ...")
    _write_query_audit(out / "e1_cser_query_audit.jsonl", ctx_te, e1_results,
                       args.budget)
    mask_dist = _write_mask_distribution(out / "selector_mask_distribution.json",
                                         ctx_te, e1_results)
    expert_delta = _write_expert_delta_summary(out / "expert_delta_summary.json",
                                               ctx_te, e1_results, args.budget)
    print("[E3] conformal coverage ...")
    e3 = exp_e3(ctx_cal, ctx_te)
    (out / "e3_conformal.json").write_text(json.dumps(e3, indent=2, default=str))
    print("[E4] budget curve ...")
    e4 = exp_e4(ctx_te, model, gate, candidate_top_k=args.candidate_top_k)
    (out / "e4_budget_curve.json").write_text(json.dumps(e4, indent=2, default=str))
    print("[E5] SVN ablation ...")
    e5 = exp_e5(
        oracle_tr,
        ctx_te,
        gate,
        args.budget,
        args.epochs,
        args.seed,
        args.candidate_top_k,
        args.train_device,
        args.train_batch_size,
    )
    (out / "e5_svn_ablation.json").write_text(json.dumps(e5, indent=2, default=str))
    print("[E6] safety ablation ...")
    e6 = exp_e6(ctx_cal, ctx_te, model, args.budget, args.alpha,
                args.candidate_top_k, selector=selector)
    (out / "e6_safety_ablation.json").write_text(json.dumps(e6, indent=2, default=str))
    top_k_sweep = None
    if candidate_top_k_list is not None:
        print("[sweep] candidate_top_k ...")
        top_k_sweep = _candidate_top_k_sweep(
            ctx_te, model, selector, args.budget, split_gate, mond_gate,
            heuristic_gate, candidate_top_k_list)
        (out / "candidate_top_k_sweep.json").write_text(
            json.dumps(top_k_sweep, indent=2, default=str))

    summary = {
        "source": source, "metric": args.metric, "budget": args.budget,
        "alpha": args.alpha, "real_models": args.real_models,
        "candidate_top_k": e1_candidate_top_k,
        "candidate_top_k_sweep": candidate_top_k_list,
        "selector": args.selector,
        "min_delta": args.min_delta,
        "expert_roster": args.expert_roster,
        "safety_config": args.safety_config,
        "safety_mode": safety_mode,
        "cost_kind": "offline_index_expert_unit_proxy",
        "gallery_size": ds.gallery_size,
        "n_videos_total": int(ds.n_videos_total),
        "n_videos_loaded": int(ds.gallery_size),
        "failed_video_ids": list(ds.failed_video_ids),
        "gallery_cache_manifest": ds.cache_manifest,
        "n_queries": ds.n_queries,
        "split": {"train": int(len(tr_idx)), "cal": int(len(cal_idx)),
                  "test": int(len(te_idx))},
        "e1_main_results": e1,
        "selector_mask_distribution": mask_dist,
        "expert_delta_summary": expert_delta,
        "candidate_top_k_sweep_results": top_k_sweep,
        "production_gate": gate.to_dict() if gate is not None else None,
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
