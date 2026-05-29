"""Benchmark runner: runs multiple strategies over multiple videos."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

from baselines.independent import IndependentBaseline
from baselines.union_fps import UnionFpsBaseline
from core.framework import LiteVTRFramework
from .metrics import compute_accuracy_vs_oracle
from .reporter import generate_tables


def _build_strategies(tasks_factory: Callable):
    """Return dict of {strategy_name: engine factory}."""
    return {
        "A_independent":    lambda: IndependentBaseline(tasks_factory()),
        "B_union_fps":      lambda: UnionFpsBaseline(tasks_factory()),
        "C_framework":      lambda: LiteVTRFramework(
            tasks_factory(),
            enable_two_stage=True,
            enable_prefilter=True,
        ),
        "C1_no_prefilter":  lambda: LiteVTRFramework(
            tasks_factory(),
            enable_two_stage=True,
            enable_prefilter=False,
        ),
        "C2_no_two_stage":  lambda: LiteVTRFramework(
            tasks_factory(),
            enable_two_stage=False,
            enable_prefilter=True,
        ),
    }


class BenchmarkRunner:
    """Run all strategies across all videos, collect metrics, emit tables."""

    def __init__(
        self,
        videos: List[Dict],
        tasks_factory: Callable,
        output_dir: str = ".",
        oracle_strategy: str = "A_independent",
    ) -> None:
        self.videos = videos
        self.tasks_factory = tasks_factory
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.oracle_strategy = oracle_strategy

    # ------------------------------------------------------------------

    def run_all(
        self,
        report_path: str = "BENCHMARK_REPORT.md",
        raw_path: str = "benchmark_raw.json",
    ) -> Dict:
        strategies = _build_strategies(self.tasks_factory)
        all_rows: Dict[str, List[dict]] = {name: [] for name in strategies}

        # Step 1: oracle first (for accuracy comparison).
        oracle_cache: Dict[str, Dict] = {}
        print(f"\n[Oracle] Running '{self.oracle_strategy}' as accuracy "
              f"reference...\n")
        for video in self.videos:
            oracle = strategies[self.oracle_strategy]()
            t0 = time.perf_counter()
            results = oracle.run(
                video_path=video["path"],
                duration=video["duration"],
                video_id=video["id"],
                sensor_stream=video.get("sensor"),
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            oracle_cache[video["id"]] = results
            row = self._row_for(
                video, self.oracle_strategy, oracle, results,
                oracle_cache[video["id"]], wall_ms
            )
            all_rows[self.oracle_strategy].append(row)
            print(f"  [{video['id']}] oracle: "
                  f"{wall_ms:.0f}ms / {row['decoded_frames']} frames")

        # Step 2: all other strategies.
        for name, factory in strategies.items():
            if name == self.oracle_strategy:
                continue
            print(f"\n[Strategy] {name}\n")
            for video in self.videos:
                engine = factory()
                t0 = time.perf_counter()
                results = engine.run(
                    video_path=video["path"],
                    duration=video["duration"],
                    video_id=video["id"],
                    sensor_stream=video.get("sensor"),
                )
                wall_ms = (time.perf_counter() - t0) * 1000.0
                row = self._row_for(
                    video, name, engine, results,
                    oracle_cache[video["id"]], wall_ms,
                )
                all_rows[name].append(row)
                print(f"  [{video['id']}] {name}: {wall_ms:.0f}ms / "
                      f"{row['decoded_frames']} frames / "
                      f"acc={row.get('acc_avg', 0.0):.3f}")

        # Step 3: save raw + tables.
        raw_out = self.out / raw_path
        with open(raw_out, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, default=str)
        print(f"\n[Raw results] {raw_out}")

        report_out = self.out / report_path
        generate_tables(all_rows, out_md=str(report_out))
        return all_rows

    # ------------------------------------------------------------------

    def _row_for(self, video, strategy_name, engine, results,
                 oracle_results, wall_ms) -> dict:
        stats = getattr(engine, "stats", {}) or {}
        acc = compute_accuracy_vs_oracle(results, oracle_results)
        # acc_avg only from [0,1] metrics (_acc, _seg_iou); skip _bnd_mae
        acc_vals = [v for k, v in acc.items()
                    if k.endswith("_acc") or k.endswith("_seg_iou")]
        acc_avg = (
            sum(acc_vals) / len(acc_vals) if acc_vals else 0.0
        )
        row = {
            "video_id": video["id"],
            "duration_sec": video["duration"],
            "strategy": strategy_name,
            "wall_ms": wall_ms,
            "decoded_frames": int(stats.get("total_decoded_frames", 0)),
            "acc_avg": acc_avg,
        }
        # breakdowns
        for k, v in stats.items():
            if k.endswith("_ms") or k.startswith("n_") or k in (
                "cache_hits", "cache_misses", "total_interval_sec"
            ):
                row[f"stat_{k}"] = v
        for k, v in acc.items():
            row[k] = v
        # per-task frame counts
        for tid, r in results.items():
            for mk, mv in (r.metrics or {}).items():
                row[f"{tid}_{mk}"] = mv
        return row
