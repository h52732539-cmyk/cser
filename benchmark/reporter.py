"""Format benchmark results into three markdown tables.

Table 1: Overall summary (one row per strategy)
Table 2: Per-video breakdown for the framework strategy
Table 3: Efficiency-Accuracy tradeoff ranking
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from tabulate import tabulate


def _mean(rows: List[dict], key: str, default: float = 0.0) -> float:
    vals = [float(r.get(key, default)) for r in rows if key in r]
    return float(np.mean(vals)) if vals else float(default)


def _format_pp(v: float) -> str:
    return f"{v * 100:+.2f}pp"


def generate_tables(
    results: Dict[str, List[dict]],
    out_md: str = "BENCHMARK_REPORT.md",
) -> None:
    if not results:
        print("[reporter] empty results")
        return

    strategy_names = list(results.keys())
    all_acc_keys = set()
    all_seg_keys = set()
    for rows in results.values():
        for r in rows:
            for k in r:
                if k.endswith("_acc"):
                    all_acc_keys.add(k)
                elif k.endswith("_seg_iou") or k.endswith("_bnd_mae"):
                    all_seg_keys.add(k)
    acc_keys = sorted(all_acc_keys)
    seg_keys = sorted(all_seg_keys)

    # --- Table 1: Overall summary ---------------------------------
    summary_rows = []
    for name in strategy_names:
        rows = results[name]
        if not rows:
            continue
        wall = _mean(rows, "wall_ms")
        frames = _mean(rows, "decoded_frames")
        acc_avg = _mean(rows, "acc_avg")
        per_task_acc = {k: _mean(rows, k) for k in acc_keys}
        per_seg = {k: _mean(rows, k) for k in seg_keys}
        summary_rows.append({
            "Strategy": name,
            "Avg Wall (ms)": f"{wall:.0f}",
            "Avg Frames": f"{frames:.0f}",
            "Avg Accuracy": f"{acc_avg:.3f}",
            **{k.replace("_acc", ""): f"{per_task_acc[k]:.3f}" for k in acc_keys},
            **{k: f"{per_seg[k]:.3f}" for k in seg_keys},
            "_wall_raw": wall,
            "_frames_raw": frames,
            "_acc_raw": acc_avg,
            "_per_task": per_task_acc,
        })

    if not summary_rows:
        print("[reporter] no rows to report")
        return

    # Pick A_independent as baseline (wall & accuracy ref), else the first.
    baseline = next(
        (r for r in summary_rows if r["Strategy"] == "A_independent"),
        summary_rows[0],
    )
    bl_wall = baseline["_wall_raw"]
    bl_frames = baseline["_frames_raw"]
    bl_per_task = baseline["_per_task"]

    for row in summary_rows:
        speedup = bl_wall / max(row["_wall_raw"], 1e-6)
        reduced = 1.0 - (row["_frames_raw"] / max(bl_frames, 1e-6))
        row["Speedup"] = f"{speedup:.2f}x"
        row["Frames-"] = f"{reduced * 100:.1f}%"
        for k in acc_keys:
            delta = row["_per_task"].get(k, 0.0) - bl_per_task.get(k, 0.0)
            row[f"Δ{k.replace('_acc', '')}"] = _format_pp(delta)

    # Clean raw fields for display
    for row in summary_rows:
        for k in list(row.keys()):
            if k.startswith("_"):
                del row[k]

    # --- Table 2: Per-video framework breakdown --------------------
    pv_rows: List[dict] = []
    fw_rows = results.get("C_framework", [])
    for r in fw_rows:
        pv_rows.append({
            "video": r["video_id"],
            "duration": f"{r['duration_sec']:.1f}s",
            "prefilter (ms)": f"{r.get('stat_prefilter_ms', 0):.0f}",
            "stage1_decode (ms)": f"{r.get('stat_stage1_decode_ms', 0):.0f}",
            "stage1_compute (ms)": f"{r.get('stat_stage1_compute_ms', 0):.0f}",
            "stage2_decode (ms)": f"{r.get('stat_stage2_decode_ms', 0):.0f}",
            "stage2_compute (ms)": f"{r.get('stat_stage2_compute_ms', 0):.0f}",
            "total (ms)": f"{r['wall_ms']:.0f}",
            "frames": int(r.get("decoded_frames", 0)),
            "S1 frames": int(r.get("stat_n_stage1_frames", 0)),
            "S2 frames": int(r.get("stat_n_stage2_frames", 0)),
            "intervals": int(r.get("stat_n_intervals", 0)),
        })

    # --- Table 3: Efficiency-Accuracy tradeoff ---------------------
    eff_rows: List[dict] = []
    for name in strategy_names:
        rows = results[name]
        if not rows:
            continue
        wall = _mean(rows, "wall_ms")
        acc = _mean(rows, "acc_avg")
        eff = acc / max(wall / 1000.0, 1e-6)
        eff_rows.append({
            "Strategy": name,
            "Avg Wall (ms)": f"{wall:.0f}",
            "Avg Accuracy": f"{acc:.3f}",
            "Eff-Acc Score": f"{eff:.3f}",
        })

    # ------------- Print to stdout --------------------------------
    print("\n" + "=" * 84)
    print("Table 1: Overall Multi-Model Benchmark Summary")
    print("=" * 84)
    print(tabulate(summary_rows, headers="keys", tablefmt="github"))

    print("\n" + "=" * 84)
    print("Table 2: Per-Video Breakdown (C_framework)")
    print("=" * 84)
    if pv_rows:
        print(tabulate(pv_rows, headers="keys", tablefmt="github"))
    else:
        print("  (no framework runs to display)")

    print("\n" + "=" * 84)
    print("Table 3: Efficiency-Accuracy Tradeoff")
    print("=" * 84)
    print(tabulate(eff_rows, headers="keys", tablefmt="github"))

    # ------------- Write markdown ---------------------------------
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# LiteVTR Multi-Model Framework — Benchmark Report\n\n")
        f.write("## Table 1: Overall Multi-Model Benchmark Summary\n\n")
        f.write(tabulate(summary_rows, headers="keys", tablefmt="github"))
        f.write("\n\n## Table 2: Per-Video Breakdown (C_framework)\n\n")
        if pv_rows:
            f.write(tabulate(pv_rows, headers="keys", tablefmt="github"))
        else:
            f.write("_(no framework runs to display)_")
        f.write("\n\n## Table 3: Efficiency-Accuracy Tradeoff\n\n")
        f.write(tabulate(eff_rows, headers="keys", tablefmt="github"))
        f.write("\n\n---\n\n")
        f.write("**Legend.** Speedup = Wall_A / Wall_X. "
                "Δmetric = metric_X − metric_A (in pp). "
                "Eff-Acc = Avg Accuracy / (Avg Wall / 1000s).\n")
    print(f"\n[Report saved] {out_md}")
