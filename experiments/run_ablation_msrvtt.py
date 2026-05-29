"""Run full module ablation on MSR-VTT 1K.

Produces a markdown table showing per-module contribution and a CSV
file for paper-style figure plotting.

Usage:
    python experiments\run_ablation_msrvtt.py \
        --cache <msrvtt_cache.npz> \
        --csv   <msrvtt_test_1k.csv> \
        --precomputed-text-embs BENCHMARK_MSRVTT_V2_full.text_embs.npy \
        --suite modules                     # or 'hp_sweep' / 'all'

Suites:
  modules     full leave-one-out (P1-P3)
  hp_sweep    α/τ/col_β/topm grid search
  all         modules + hp_sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablation import (
    AblationConfig, make_module_ablation_suite, make_hp_sweep_suite, run_suite,
)
from core.meta_filter import MetaFilter

# Reuse the loaders from the v3 benchmark
from demo.run_msrvtt_meta_v3 import (
    load_index, load_queries, make_synthetic_intents,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csv",   required=True)
    ap.add_argument("--precomputed-text-embs", required=True)
    ap.add_argument("--suite", choices=("modules", "hp_sweep", "all"),
                    default="modules")
    ap.add_argument("--with-meta", action="store_true",
                    help="include synthetic-meta intents (Phase-3 ablations)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments/results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print("[load] index ...")
    index, _ = load_index(args.cache, rng)
    print(f"   N_videos={index.size}")

    print("[load] queries ...")
    queries, gt = load_queries(args.csv)
    q_embs = np.load(args.precomputed_text_embs).astype(np.float32)[:len(queries)]
    q_embs /= np.linalg.norm(q_embs, axis=-1, keepdims=True) + 1e-9

    intents = None
    mf = None
    if args.with_meta:
        intents = make_synthetic_intents(index, gt, rng)
        mf = MetaFilter(time_slack_sec=3600.0, strict=False)
        n_con = sum(1 for i in intents if i.has_constraint())
        print(f"   {n_con}/{len(intents)} queries carry meta constraints")

    base = AblationConfig(name="default")

    if args.suite in ("modules", "all"):
        print("\n=== MODULE ABLATION ===")
        suite = make_module_ablation_suite(base)
        results = run_suite(
            suite, index, q_embs, gt,
            intents=intents, meta_filter=mf,
            save_path=str(out_dir / "ablation_modules.json"),
        )
        _print_module_table(results, out_dir)

    if args.suite in ("hp_sweep", "all"):
        print("\n=== HP SWEEP ===")
        suite = make_hp_sweep_suite(base)
        print(f"   {len(suite)} configurations")
        results = run_suite(
            suite, index, q_embs, gt,
            intents=intents, meta_filter=mf,
            save_path=str(out_dir / "ablation_hp_sweep.json"),
            verbose=False,
        )
        _print_hp_pareto(results, out_dir)


def _print_module_table(results: List[dict], out_dir: Path) -> None:
    full = next(r for r in results if r["name"] == "A0_full")
    rows = []
    for r in results:
        rows.append({
            "name": r["name"],
            "R@1":  f"{r['R@1']*100:5.2f}",
            "ΔR@1": f"{(r['R@1']-full['R@1'])*100:+5.2f}",
            "R@5":  f"{r['R@5']*100:5.2f}",
            "R@10": f"{r['R@10']*100:5.2f}",
            "MeanR": f"{r['MeanR']:.1f}",
            "ms/q":  f"{r['ms_per_query']:.2f}",
        })

    print("\n--- Module ablation summary (Δ vs A0_full) ---")
    print(f"{'name':<25} {'R@1':>6} {'ΔR@1':>7} {'R@5':>6} {'R@10':>6} "
          f"{'MeanR':>7} {'ms/q':>7}")
    for r in rows:
        print(f"{r['name']:<25} {r['R@1']:>6} {r['ΔR@1']:>7} {r['R@5']:>6} "
              f"{r['R@10']:>6} {r['MeanR']:>7} {r['ms/q']:>7}")

    csv_path = out_dir / "ablation_modules.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] {csv_path}")


def _print_hp_pareto(results: List[dict], out_dir: Path) -> None:
    # Sort by R@1 descending, show top-10
    s = sorted(results, key=lambda r: -r["R@1"])
    print("\nTop-10 hyperparam configurations:")
    print(f"{'name':<32} {'R@1':>6} {'R@5':>6} {'MeanR':>7} {'ms/q':>7}")
    for r in s[:10]:
        print(f"{r['name']:<32} {r['R@1']*100:5.2f} {r['R@5']*100:5.2f} "
              f"{r['MeanR']:7.1f} {r['ms_per_query']:7.2f}")
    csv_path = out_dir / "ablation_hp_sweep.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "alpha_nnn", "tau_qamp", "col_beta", "topm_rerank",
                    "R@1", "R@5", "R@10", "MeanR", "ms_per_query"])
        for r in results:
            cfg = r["config"]
            w.writerow([r["name"], cfg["alpha_nnn"], cfg["tau_qamp"],
                        cfg["col_beta"], cfg["topm_rerank"],
                        r["R@1"], r["R@5"], r["R@10"],
                        r["MeanR"], r["ms_per_query"]])
    print(f"\n[saved] {csv_path}")


if __name__ == "__main__":
    main()
