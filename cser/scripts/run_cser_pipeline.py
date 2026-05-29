"""Run the CSER training/calibration/evaluation pipeline.

With no dataset arguments this runs a deterministic synthetic pipeline. With
`--cache`, `--csv`, and `--text-embs`, it uses existing MSR-VTT-style inputs
and a CSER expert cache derived from the semantic archive.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from cser.baselines import RandomSubsetPolicy, all_experts, fixed_cascade_subset, semantic_only
from cser.conformal import MondrianConformalCalibrator, gt_indices_from_ids
from cser.expert_store import ExpertOutputStore
from cser.features import build_query_features
from cser.labels import build_oracle_labels
from cser.metrics import submodularity_violation_rate, summarize_results
from cser.planner import GreedyBudgetedSelector
from cser.subset_executor import CSERSubsetExecutor

try:
    from cser.train_svn import SVNTrainConfig, train_svn
except ImportError:  # torch is optional for smoke/mock-only environments
    SVNTrainConfig = None
    train_svn = None


class MeanMarginalModel:
    """Tiny fallback policy used when torch is not installed."""

    def __init__(self, labels) -> None:
        flat = labels.marginal_values.reshape(-1, labels.n_experts)
        vals = np.zeros(labels.n_experts, dtype=np.float32)
        for i in range(labels.n_experts):
            finite = flat[:, i][np.isfinite(flat[:, i])]
            vals[i] = float(finite.mean()) if finite.size else 0.0
        self.values = vals

    def __call__(self, query_features, selected_mask):
        return self.values.copy()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _load_csv_queries(csv_path: str | Path) -> Tuple[List[str], List[str]]:
    queries: List[str] = []
    gt: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            queries.append(row.get("sentence") or row.get("query") or "")
            gt.append(row["video_id"])
    return queries, gt


def _make_synthetic_queries(
    store: ExpertOutputStore,
    n_queries: int,
    seed: int,
) -> Tuple[np.ndarray, List[str], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    gt_idx = rng.integers(0, store.size, size=n_queries)
    gt_ids = [store.video_ids[int(i)] for i in gt_idx]
    noise = rng.normal(scale=0.12, size=(n_queries, store.dim)).astype(np.float32)
    query_embs = _normalize_rows(store.clip_video_embs[gt_idx] + noise)
    contexts = [store.query_context_for_gt(g, rng) for g in gt_ids]
    return query_embs.astype(np.float32), gt_ids, contexts


def _semantic_matrix(executor: CSERSubsetExecutor, query_embs: np.ndarray) -> np.ndarray:
    return np.stack([executor.semantic_scores(q) for q in query_embs]).astype(np.float32)


def _features_for(
    executor: CSERSubsetExecutor,
    query_embs: np.ndarray,
    contexts: Sequence[Dict[str, object]],
    budget: float,
) -> np.ndarray:
    rows = []
    for q, ctx in zip(query_embs, contexts):
        rows.append(build_query_features(q, executor.semantic_scores(q), ctx, budget=budget))
    return np.stack(rows).astype(np.float32)


def _split(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(0.60 * n))
    n_cal = max(1, int(0.20 * n))
    train = perm[:n_train]
    cal = perm[n_train : n_train + n_cal]
    test = perm[n_train + n_cal :]
    if test.size == 0:
        test = cal
    return train, cal, test


def _evaluate_subset_policy(
    name: str,
    executor: CSERSubsetExecutor,
    query_embs: np.ndarray,
    gt_ids: Sequence[str],
    contexts: Sequence[Dict[str, object]],
    subsets: Sequence[Sequence[str]],
) -> Dict[str, object]:
    ranks = []
    filtered = []
    costs = []
    for q, gt, ctx, subset in zip(query_embs, gt_ids, contexts, subsets):
        res = executor.execute_subset(subset, q, gt, query_context=ctx)
        ranks.append(res.rank)
        filtered.append(res.gt_filtered)
        costs.append(res.cost)
    return summarize_results(name, ranks, filtered, costs)


def _evaluate_cser(
    selector: GreedyBudgetedSelector,
    executor: CSERSubsetExecutor,
    features: np.ndarray,
    query_embs: np.ndarray,
    gt_ids: Sequence[str],
    contexts: Sequence[Dict[str, object]],
    budget: float,
) -> Tuple[Dict[str, object], Dict[str, int]]:
    ranks = []
    filtered = []
    costs = []
    conformal_sizes = []
    selected_counts: Dict[str, int] = {}
    for feat, q, gt, ctx in zip(features, query_embs, gt_ids, contexts):
        decision, result = selector.plan_and_execute(
            feat, q, gt, budget=budget, executor=executor, query_context=ctx
        )
        ranks.append(result.rank)
        filtered.append(result.gt_filtered)
        costs.append(result.cost)
        conformal_sizes.append(decision.conformal_set_size)
        key = "+".join(decision.selected_experts)
        selected_counts[key] = selected_counts.get(key, 0) + 1
    return summarize_results("CSER_mondrian", ranks, filtered, costs, conformal_sizes), selected_counts


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache:
        store = ExpertOutputStore.from_msrvtt_cache(args.cache, seed=args.seed)
        queries, gt_ids = _load_csv_queries(args.csv)
        query_embs = np.load(args.text_embs).astype(np.float32)[: len(gt_ids)]
        query_embs = _normalize_rows(query_embs)
        rng = np.random.default_rng(args.seed)
        contexts = [store.query_context_for_gt(g, rng) for g in gt_ids]
    else:
        store = ExpertOutputStore.synthetic(n_videos=args.synthetic_videos, seed=args.seed)
        query_embs, gt_ids, contexts = _make_synthetic_queries(
            store, args.synthetic_queries, args.seed + 1
        )

    store.save(out_dir / "expert_cache.npz")
    executor = CSERSubsetExecutor(store)
    train_idx, cal_idx, test_idx = _split(len(gt_ids), args.seed)

    def take(arr, idx):
        if isinstance(arr, np.ndarray):
            return arr[idx]
        return [arr[int(i)] for i in idx]

    train_embs = take(query_embs, train_idx)
    train_gt = take(gt_ids, train_idx)
    train_ctx = take(contexts, train_idx)
    cal_embs = take(query_embs, cal_idx)
    cal_gt = take(gt_ids, cal_idx)
    cal_ctx = take(contexts, cal_idx)
    test_embs = take(query_embs, test_idx)
    test_gt = take(gt_ids, test_idx)
    test_ctx = take(contexts, test_idx)

    train_labels = build_oracle_labels(executor, train_embs, train_gt, train_ctx)
    train_labels.save(out_dir / "oracle_labels_train.npz")
    train_features = _features_for(executor, train_embs, train_ctx, budget=args.budget)

    if train_svn is None or SVNTrainConfig is None:
        model = MeanMarginalModel(train_labels)
        history = {
            "warning": "torch is not installed; using mean marginal fallback model",
            "n_examples": int(np.isfinite(train_labels.marginal_values).sum()),
        }
    else:
        model, history = train_svn(
            train_features,
            train_labels,
            SVNTrainConfig(epochs=args.epochs, batch_size=args.batch_size, patience=10, seed=args.seed),
            save_dir=out_dir / "model",
            verbose=bool(args.verbose),
        )

    cal_scores = _semantic_matrix(executor, cal_embs)
    cal_gt_idx = gt_indices_from_ids(store.video_ids, cal_gt)
    conformal = MondrianConformalCalibrator(
        alpha=args.alpha,
        n_bins=args.mondrian_bins,
        min_bin_size=args.min_bin_size,
    ).fit(cal_scores, cal_gt_idx)

    test_scores = _semantic_matrix(executor, test_embs)
    test_gt_idx = gt_indices_from_ids(store.video_ids, test_gt)
    coverage_report = conformal.report(test_scores, test_gt_idx)

    random_policy = RandomSubsetPolicy(seed=args.seed)
    subsets_sem = [("clip_semantic",) for _ in test_gt]
    subsets_all = [("clip_semantic", "face_detect", "arcface", "highlight", "scene") for _ in test_gt]
    subsets_cascade = [fixed_cascade_subset(args.budget) for _ in test_gt]
    subsets_random = [random_policy.select(args.budget) for _ in test_gt]

    results = [
        _evaluate_subset_policy("semantic_only", executor, test_embs, test_gt, test_ctx, subsets_sem),
        _evaluate_subset_policy("all_experts", executor, test_embs, test_gt, test_ctx, subsets_all),
        _evaluate_subset_policy("fixed_cascade", executor, test_embs, test_gt, test_ctx, subsets_cascade),
        _evaluate_subset_policy("random_subset", executor, test_embs, test_gt, test_ctx, subsets_random),
    ]

    test_features = _features_for(executor, test_embs, test_ctx, budget=args.budget)
    selector = GreedyBudgetedSelector(model, conformal_gate=conformal, tau_stop=args.tau_stop)
    cser_result, selected_counts = _evaluate_cser(
        selector, executor, test_features, test_embs, test_gt, test_ctx, args.budget
    )
    results.append(cser_result)

    submod = {
        "violation_rate": submodularity_violation_rate(
            train_labels.marginal_values, train_labels.subset_masks
        )
    }
    payload = {
        "split": {"train": int(len(train_idx)), "cal": int(len(cal_idx)), "test": int(len(test_idx))},
        "history": history,
        "coverage": coverage_report,
        "submodularity": submod,
        "expert_selection": selected_counts,
        "results": results,
    }

    (out_dir / "main_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "coverage.json").write_text(json.dumps(coverage_report, indent=2), encoding="utf-8")
    (out_dir / "submodularity.json").write_text(json.dumps(submod, indent=2), encoding="utf-8")
    (out_dir / "expert_selection.json").write_text(json.dumps(selected_counts, indent=2), encoding="utf-8")
    if results:
        keys = sorted({k for row in results for k in row.keys()})
        with open(out_dir / "main_results.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--text-embs", default=None)
    parser.add_argument("--out-dir", default="reports/cser")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tau-stop", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mondrian-bins", type=int, default=3)
    parser.add_argument("--min-bin-size", type=int, default=30)
    parser.add_argument("--synthetic-videos", type=int, default=128)
    parser.add_argument("--synthetic-queries", type=int, default=120)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if bool(args.cache) != bool(args.csv) or bool(args.cache) != bool(args.text_embs):
        raise SystemExit("--cache, --csv, and --text-embs must be provided together")

    payload = run_pipeline(args)
    print(json.dumps(payload["results"], indent=2))
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
