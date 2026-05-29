"""Build a CSER expert cache."""
from __future__ import annotations

import argparse
from pathlib import Path

from cser.expert_store import ExpertOutputStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=None, help="Existing MSR-VTT npz cache")
    parser.add_argument("--out", default="reports/cser/expert_cache.npz")
    parser.add_argument("--synthetic-videos", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.cache:
        store = ExpertOutputStore.from_msrvtt_cache(args.cache, seed=args.seed)
    else:
        store = ExpertOutputStore.synthetic(n_videos=args.synthetic_videos, seed=args.seed)
    store.save(args.out)
    print(f"[saved] {Path(args.out)}  n_videos={store.size}")


if __name__ == "__main__":
    main()
