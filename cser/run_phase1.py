"""Phase-1 driver: data -> oracle labels -> train SVN -> verify submodularity.

Uses the 5 real expert models (mock fallback) over a video gallery.

    # framework smoke / synthetic (no external files, mock experts)
    python -m cser.run_phase1 --out-dir reports/cser_phase1

    # real expert models over a real video gallery
    python -m cser.run_phase1 --out-dir reports/cser_phase1 \
        --videos /path/to/videos_dir --csv /path/to/queries.csv --real-models

Produces under --out-dir:
    oracle_train.npz / oracle_test.npz   exact value lattices
    svn/svn.pt + svn_config.json         trained Submodular Value Network
    submodularity_report.json            E2 result (data-level + SVN-level)
    greedy_vs_oracle.json                SVN-greedy value vs oracle ceiling
    phase1_summary.json                  roll-up + verdict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cser.data import build_synthetic_dataset, load_video_dataset
from cser.retrieval import RetrievalEngine
from cser.value_oracle import build_oracle_labels, OracleLabels
from cser.train_svn import train_svn, SVNTrainConfig
from cser.svn import SubmodularValueNetwork
from cser.greedy import GreedyBudgetedSelector
from cser.submodularity import verify_submodularity
from cser.experts import (N_OPTIONAL, OPTIONAL_NAMES, mask_to_id, selection_cost,
                          all_optional_masks)


def _svn_prediction_submod_violation(model, feats: np.ndarray) -> float:
    model.eval()
    x = torch.from_numpy(feats.astype(np.float32))
    B = x.shape[0]
    with torch.no_grad():
        base = model(x, torch.zeros(B, N_OPTIONAL)).numpy()
        viol, total = 0, 0
        for ep in range(N_OPTIONAL):
            m = torch.zeros(B, N_OPTIONAL); m[:, ep] = 1.0
            v = model(x, m).numpy()
            for j in range(N_OPTIONAL):
                if j == ep:
                    continue
                viol += int((v[:, j] - base[:, j] > 1e-4).sum())
                total += B
    return viol / max(total, 1)


def _greedy_vs_oracle(model, oracle: OracleLabels, budgets=(3.0, 5.0, 9.5)):
    out = {}
    best = oracle.best_subset_value()
    empty = oracle.empty_set_value()
    for B in budgets:
        sel = GreedyBudgetedSelector(model, budget=B)
        realised = np.zeros(oracle.n_queries)
        ncalls = np.zeros(oracle.n_queries)
        for q in range(oracle.n_queries):
            r = sel.select(oracle.query_feats[q])
            realised[q] = oracle.value_matrix[q, mask_to_id(r.selected_mask)]
            ncalls[q] = r.n_experts_called
        denom = max(float(best.mean()), 1e-9)
        out[f"budget={B:.1f}"] = {
            "greedy_value_mean": float(realised.mean()),
            "oracle_value_mean": float(best.mean()),
            "semantic_only_value_mean": float(empty.mean()),
            "pct_of_oracle": float(realised.mean() / denom),
            "avg_experts_called": float(ncalls.mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/cser_phase1")
    ap.add_argument("--videos", default=None, help="real video gallery dir")
    ap.add_argument("--csv", default=None, help="real queries csv")
    ap.add_argument("--real-models", action="store_true",
                    help="use real expert backbones (needs weights)")
    ap.add_argument("--metric", default="rr",
                    choices=["rr", "recall@1", "recall@5", "recall@10"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--variant", default="full",
                    choices=["full", "no_cross_attn", "no_set_conditioning"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--syn-videos", type=int, default=80)
    ap.add_argument("--syn-queries", type=int, default=160)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ── 1. Data (runs the 5 experts over the gallery once) ──
    if args.videos:
        print(f"[1/5] Loading real video gallery from {args.videos}")
        ds = load_video_dataset(args.videos, args.csv,
                                use_real_models=args.real_models, seed=args.seed)
        source = "video"
    else:
        print("[1/5] Building synthetic gallery (mock experts, no external files)")
        ds = build_synthetic_dataset(n_videos=args.syn_videos,
                                     n_queries=args.syn_queries,
                                     use_real_models=args.real_models,
                                     seed=args.seed)
        source = "synthetic"
    print(f"      source={source} gallery={ds.gallery_size} queries={ds.n_queries}")

    tr_idx, cal_idx, te_idx = ds.split(seed=args.seed)
    engine = RetrievalEngine(ds.gallery)

    def _sub(idx):
        return ([ds.query_priors[i] for i in idx],
                [ds.gt_video_ids[i] for i in idx])

    # ── 2. Oracle labels ──
    print(f"[2/5] Building oracle value lattices (metric={args.metric}) ...")
    p_tr, g_tr = _sub(tr_idx)
    p_te, g_te = _sub(te_idx)
    oracle_tr = build_oracle_labels(engine, p_tr, g_tr, metric=args.metric, verbose=False)
    oracle_te = build_oracle_labels(engine, p_te, g_te, metric=args.metric, verbose=False)
    oracle_tr.save(str(out / "oracle_train.npz"))
    oracle_te.save(str(out / "oracle_test.npz"))

    # ── 3. Train SVN ──
    print("[3/5] Training Submodular Value Network ...")
    cfg = SVNTrainConfig(epochs=args.epochs, variant=args.variant, seed=args.seed)
    model, history = train_svn(oracle_tr, cfg, save_dir=str(out / "svn"))

    # ── 4. Submodularity verification (E2) ──
    print("[4/5] Verifying submodularity (E2) ...")
    report = verify_submodularity(oracle_te)
    svn_pred_viol = _svn_prediction_submod_violation(model, oracle_te.query_feats)
    rd = report.to_dict()
    rd["svn_prediction_violation_rate"] = svn_pred_viol
    (out / "submodularity_report.json").write_text(json.dumps(rd, indent=2))
    print(f"      verdict={report.verdict}  "
          f"submod_violation={report.submodularity_violation_rate:.3%}  "
          f"gamma_mean={report.gamma_ratio_mean:.3f}  "
          f"gamma_p10={report.gamma_ratio_p10:.3f}")

    # ── 5. Greedy vs oracle ──
    print("[5/5] SVN-greedy vs oracle ceiling ...")
    gvo = _greedy_vs_oracle(model, oracle_te)
    (out / "greedy_vs_oracle.json").write_text(json.dumps(gvo, indent=2))
    for k, v in gvo.items():
        print(f"      {k}: greedy={v['greedy_value_mean']:.3f} "
              f"({v['pct_of_oracle']:.1%} of oracle) "
              f"experts={v['avg_experts_called']:.2f}")

    summary = {
        "source": source, "metric": args.metric, "svn_variant": args.variant,
        "real_models": args.real_models, "gallery_size": ds.gallery_size,
        "n_queries": ds.n_queries,
        "split": {"train": int(len(tr_idx)), "cal": int(len(cal_idx)),
                  "test": int(len(te_idx))},
        "svn_param_count": model.param_count(),
        "svn_best_val_mse": history["val_mse"][-1] if history["val_mse"] else None,
        "submodularity": rd, "greedy_vs_oracle": gvo,
        "optional_experts": list(OPTIONAL_NAMES),
    }
    (out / "phase1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] artifacts in {out}/")
    print(f"       submodularity verdict: {report.verdict.upper()}")


if __name__ == "__main__":
    main()
