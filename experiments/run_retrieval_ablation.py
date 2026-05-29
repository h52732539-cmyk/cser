"""Retrieval ablation — modules that affect retrieval scoring.

Operates on pre-computed MSR-VTT CLIP embeddings. Toggles only the
modules that actually take effect in the embedding-space retrieval
pipeline:

    R0  full pipeline
    R1  no Multi-K  (use only K=6 protos, skip K=2/4 averaging)
    R2  no NNN      (alpha_nnn = 0)
    R3  no QAMP     (treat protos as max-pooled, skip softmax)
    R4  no col-softmax (col_beta = 0)
    R5  no rerank   (topm_rerank = 0, raw cosine only)
    R6  no offline-index (= raw cosine baseline, equivalent to R5+R1+R3)

    M1  no meta-filter  (Phase 3 hard filter off)
    M2  no meta-fusion  (Phase 3 soft α blending off)
    M3  no metadata at all (M1 + M2)

Hyperparameter sweeps:
    H_alpha     α_nnn ∈ {0.3, 0.5, 0.7, 0.9}
    H_tau       τ ∈ {0.01, 0.02, 0.05, 0.10}
    H_col_beta  β ∈ {0.0, 0.2, 0.4, 0.6}
    H_topm      topM ∈ {50, 100, 300, 500}
    H_meta      α_meta ∈ {0.3, 0.5, 0.7, 0.9}

Each config is a *named recipe* that calls into search_batch /
search_with_meta with the right flags so the table shows real Δs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.offline_index import OfflineIndex
from core.meta_filter import MetaFilter
from core.query_parser import QueryIntent

# Reuse loaders
from demo.run_msrvtt_meta_v3 import (
    load_index, load_queries, make_synthetic_intents,
)


# ----------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    name: str = "default"
    use_multi_k:        bool = True
    use_nnn:            bool = True
    use_qamp:           bool = True
    use_col_softmax:    bool = True
    use_rerank:         bool = True
    use_offline_index:  bool = True
    use_meta_filter:    bool = True
    use_meta_fusion:    bool = False     # Phase-3 default OFF (ablation showed harmful)
    alpha_nnn:    float = 0.7            # joint-optimum from MSR-VTT sweep
    tau_qamp:     float = 0.10
    col_beta:     float = 0.4
    topm_rerank:  int   = 500
    meta_alpha:   float = 1.0            # 1.0 = pure semantic in fused term

    def to_dict(self) -> Dict:
        return asdict(self)


# ----------------------------------------------------------------------

def evaluate_one(cfg: RetrievalConfig,
                  index: OfflineIndex,
                  q_embs: np.ndarray,
                  gt: List[str],
                  intents: Optional[List[QueryIntent]] = None,
                  meta_filter: Optional[MetaFilter] = None,
                  ) -> Dict:
    N = len(gt)
    t0 = time.perf_counter()

    # --- Determine effective hyperparameters from flags ---
    alpha = cfg.alpha_nnn if cfg.use_nnn else 0.0
    tau   = cfg.tau_qamp  # tau is always used by QAMP softmax; turning
                          # off QAMP is handled by replacing the call below.
    col_b = cfg.col_beta if cfg.use_col_softmax else 0.0
    topm  = cfg.topm_rerank if cfg.use_rerank else 0

    if not cfg.use_offline_index:
        # Raw cosine baseline — bypass everything.
        big = index._flat_protos[max(index._flat_protos.keys())]
        sl = index._flat_slices_by_k[max(index._flat_slices_by_k.keys())]
        sims_all = q_embs @ big.T
        scores = np.full((N, len(index.entries)), -1e9, dtype=np.float32)
        for j, (s, e) in enumerate(sl):
            if e > s:
                scores[:, j] = sims_all[:, s:e].max(axis=1)
        ranks = []
        for i in range(N):
            order = np.argsort(-scores[i])
            ids = [index.entries[k].video_id for k in order]
            ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)
    elif intents is not None and (cfg.use_meta_filter or cfg.use_meta_fusion):
        # Phase-3 batch hybrid path (preserves col-softmax)
        all_hits = index.search_batch_with_meta(
            q_embs, intents,
            top_k=len(index.entries),
            alpha_nnn=alpha,
            tau_qamp=tau,
            col_beta=col_b,
            topm_rerank=topm if topm > 0 else 1,
            meta_filter=meta_filter,
            meta_alpha=cfg.meta_alpha if cfg.use_meta_fusion else 1.0,
            use_hard_filter=cfg.use_meta_filter,
            use_meta_fusion=cfg.use_meta_fusion,
        )
        ranks = []
        for i in range(N):
            ids = [h[0] for h in all_hits[i] if h[1] > -1e8]
            ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)
    else:
        # Phase-2 path: search_batch. To turn off Multi-K we simulate
        # by passing use_multi_k=False inside search_batch.
        all_hits = index.search_batch(
            q_embs,
            top_k=len(index.entries),
            alpha_nnn=alpha,
            tau_qamp=tau,
            col_beta=col_b,
            topm_rerank=topm if topm > 0 else 1,
            use_multi_k=cfg.use_multi_k,
        )
        ranks = []
        for i in range(N):
            ids = [h[0] for h in all_hits[i]]
            ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)

    # If QAMP disabled — replace tau by very large (≈ uniform mean) so
    # QAMP softmax degrades to a max-pool. We approximate that by
    # rerunning with tau → ∞.
    if cfg.use_offline_index and not cfg.use_qamp:
        all_hits = index.search_batch(
            q_embs,
            top_k=len(index.entries),
            alpha_nnn=alpha,
            tau_qamp=1e9,                     # ≈ flat softmax → mean-pool
            col_beta=col_b,
            topm_rerank=topm if topm > 0 else 1,
            use_multi_k=cfg.use_multi_k,
        )
        ranks = []
        for i in range(N):
            ids = [h[0] for h in all_hits[i]]
            ranks.append(ids.index(gt[i]) if gt[i] in ids else 1000)

    dt = (time.perf_counter() - t0) * 1000.0
    ranks = np.array(ranks)
    return {
        "name": cfg.name,
        "config": cfg.to_dict(),
        "R@1":   float((ranks == 0).mean()),
        "R@5":   float((ranks <  5).mean()),
        "R@10":  float((ranks < 10).mean()),
        "MedR":  float(np.median(ranks) + 1),
        "MeanR": float(ranks.mean() + 1),
        "total_ms": float(dt),
        "ms_per_query": float(dt / max(N, 1)),
    }


def it_has(it: QueryIntent) -> bool:
    return it is not None and it.has_constraint()


# ----------------------------------------------------------------------
#  Suites
# ----------------------------------------------------------------------

def make_module_suite(base: RetrievalConfig) -> List[RetrievalConfig]:
    """Per-module leave-one-out — retrieval scoring only.

    Notes:
      * M2 enables (rather than disables) meta-fusion since the
        production default keeps it off (ablation showed it harmful
        under hard-filter mode).
    """
    return [
        replace(base, name="R0_full"),
        replace(base, name="R1_no_multi_k",      use_multi_k=False),
        replace(base, name="R2_no_nnn",          use_nnn=False),
        replace(base, name="R3_no_qamp",         use_qamp=False),
        replace(base, name="R4_no_col_softmax",  use_col_softmax=False),
        replace(base, name="R5_no_rerank",       use_rerank=False),
        replace(base, name="R6_cosine_only",     use_offline_index=False),
        replace(base, name="M1_no_meta_filter",  use_meta_filter=False),
        replace(base, name="M2_with_meta_fusion",
                use_meta_fusion=True, meta_alpha=0.7),
        replace(base, name="M3_no_meta_at_all",  use_meta_filter=False,
                                                  use_meta_fusion=False),
    ]


def make_hp_suite(base: RetrievalConfig) -> List[RetrievalConfig]:
    out: List[RetrievalConfig] = []
    for a in (0.3, 0.5, 0.7, 0.9):
        out.append(replace(base, name=f"H_alpha={a}", alpha_nnn=a))
    for t in (0.01, 0.02, 0.05, 0.10):
        out.append(replace(base, name=f"H_tau={t}", tau_qamp=t))
    for c in (0.0, 0.2, 0.4, 0.6):
        out.append(replace(base, name=f"H_colbeta={c}", col_beta=c))
    for m in (50, 100, 300, 500):
        out.append(replace(base, name=f"H_topm={m}", topm_rerank=m))
    for am in (0.3, 0.5, 0.7, 0.9):
        out.append(replace(base, name=f"H_meta_alpha={am}", meta_alpha=am))
    return out


def make_joint_hp_suite(base: RetrievalConfig) -> List[RetrievalConfig]:
    """Joint-optimum scan around the per-axis best of make_hp_suite()."""
    out: List[RetrievalConfig] = []
    for a in (0.5, 0.7, 0.9):
        for t in (0.05, 0.10):
            for c in (0.4, 0.6):
                for m in (300, 500):
                    out.append(replace(
                        base,
                        name=f"J_a{a}_t{t}_c{c}_m{m}",
                        alpha_nnn=a, tau_qamp=t, col_beta=c, topm_rerank=m,
                    ))
    return out


# ----------------------------------------------------------------------

def run(suite: List[RetrievalConfig],
        index: OfflineIndex, q_embs, gt,
        intents=None, meta_filter=None,
        save: Optional[str] = None,
        verbose: bool = True) -> List[Dict]:
    rows = []
    for i, cfg in enumerate(suite):
        if verbose:
            print(f"[{i+1}/{len(suite)}] {cfg.name}")
        r = evaluate_one(cfg, index, q_embs, gt, intents, meta_filter)
        if verbose:
            print(f"   R@1={r['R@1']*100:5.2f}%  R@5={r['R@5']*100:5.2f}%  "
                  f"MeanR={r['MeanR']:.1f}  ms/q={r['ms_per_query']:.2f}")
        rows.append(r)
    if save:
        Path(save).write_text(json.dumps(rows, indent=2, default=str),
                               encoding="utf-8")
    return rows


def print_table(rows: List[Dict], full_name: str = "R0_full") -> None:
    full = next((r for r in rows if r["name"] == full_name), rows[0])
    print(f"\n{'name':<25} {'R@1':>7} {'ΔR@1':>7} {'R@5':>7} {'R@10':>7} "
          f"{'MeanR':>7} {'ms/q':>7}")
    print("-" * 75)
    for r in rows:
        d = (r["R@1"] - full["R@1"]) * 100
        print(f"{r['name']:<25} {r['R@1']*100:>6.2f}% {d:>+6.2f}  "
              f"{r['R@5']*100:>6.2f}% {r['R@10']*100:>6.2f}% "
              f"{r['MeanR']:>7.1f} {r['ms_per_query']:>7.2f}")


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv",   required=True)
    ap.add_argument("--precomputed-text-embs", required=True)
    ap.add_argument("--suite", choices=("modules", "hp", "joint", "all"),
                    default="modules")
    ap.add_argument("--with-meta", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments/results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import random
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print("[load] index ...")
    index, _ = load_index(args.cache, rng)
    print(f"   N_videos={index.size}")

    queries, gt = load_queries(args.csv)
    q_embs = np.load(args.precomputed_text_embs).astype(np.float32)[:len(queries)]
    q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9
    print(f"   N_queries={len(queries)}")

    intents = None; mf = None
    if args.with_meta:
        intents = make_synthetic_intents(index, gt, rng)
        mf = MetaFilter(time_slack_sec=3600.0, strict=False)
        n_con = sum(1 for it in intents if it.has_constraint())
        print(f"   {n_con}/{len(intents)} queries carry meta constraints")

    base = RetrievalConfig(name="default")

    if args.suite in ("modules", "all"):
        print("\n=== RETRIEVAL MODULE ABLATION ===")
        rows = run(make_module_suite(base), index, q_embs, gt,
                    intents=intents, meta_filter=mf,
                    save=str(out_dir / "retrieval_modules.json"))
        print_table(rows, "R0_full")
        # CSV
        csv_path = out_dir / "retrieval_modules.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "R@1", "ΔR@1_pp", "R@5", "R@10", "MeanR", "ms_per_query"])
            full = next((r for r in rows if r["name"] == "R0_full"), rows[0])
            for r in rows:
                w.writerow([r["name"], r["R@1"],
                             (r["R@1"] - full["R@1"]) * 100,
                             r["R@5"], r["R@10"], r["MeanR"],
                             r["ms_per_query"]])
        print(f"\n[saved] {csv_path}")

    if args.suite in ("hp", "all"):
        print("\n=== HP SWEEP ===")
        rows = run(make_hp_suite(base), index, q_embs, gt,
                    intents=intents, meta_filter=mf,
                    save=str(out_dir / "retrieval_hp.json"),
                    verbose=False)
        rows.sort(key=lambda r: -r["R@1"])
        print(f"\nTop-15 by R@1:")
        for r in rows[:15]:
            print(f"  {r['name']:<28} R@1={r['R@1']*100:5.2f}%  "
                  f"R@5={r['R@5']*100:5.2f}%  ms/q={r['ms_per_query']:.2f}")

    if args.suite in ("joint", "all"):
        print("\n=== JOINT HP SWEEP ===")
        rows = run(make_joint_hp_suite(base), index, q_embs, gt,
                    intents=intents, meta_filter=mf,
                    save=str(out_dir / "retrieval_hp_joint.json"),
                    verbose=False)
        rows.sort(key=lambda r: -r["R@1"])
        print(f"\nTop-15 joint configurations by R@1:")
        print(f"  {'name':<32} {'R@1':>6} {'R@5':>6} {'R@10':>7} {'MeanR':>7} {'ms/q':>7}")
        for r in rows[:15]:
            print(f"  {r['name']:<32} {r['R@1']*100:>5.2f}% "
                  f"{r['R@5']*100:>5.2f}% {r['R@10']*100:>6.2f}% "
                  f"{r['MeanR']:>7.1f} {r['ms_per_query']:>7.2f}")
        # save CSV for paper figures
        import csv as _csv
        csv_path = out_dir / "retrieval_hp_joint.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["name", "alpha_nnn", "tau_qamp", "col_beta",
                        "topm_rerank", "R@1", "R@5", "R@10",
                        "MeanR", "ms_per_query"])
            for r in rows:
                cfg = r["config"]
                w.writerow([r["name"], cfg["alpha_nnn"], cfg["tau_qamp"],
                            cfg["col_beta"], cfg["topm_rerank"],
                            r["R@1"], r["R@5"], r["R@10"],
                            r["MeanR"], r["ms_per_query"]])
        print(f"\n[saved] {csv_path}")


if __name__ == "__main__":
    main()
