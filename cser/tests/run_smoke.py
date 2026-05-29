"""End-to-end CSER smoke test with mock expert outputs."""
from __future__ import annotations

import json
from argparse import Namespace

from cser.scripts.run_cser_pipeline import run_pipeline


def main() -> None:
    args = Namespace(
        cache=None,
        csv=None,
        text_embs=None,
        out_dir="reports/cser/smoke",
        seed=7,
        budget=3.0,
        alpha=0.1,
        tau_stop=-1.0,
        epochs=3,
        batch_size=32,
        mondrian_bins=3,
        min_bin_size=4,
        synthetic_videos=48,
        synthetic_queries=36,
        verbose=False,
    )
    payload = run_pipeline(args)
    methods = {row["method"] for row in payload["results"]}
    assert "CSER_mondrian" in methods
    assert payload["coverage"]["avg_set_size"] >= 1
    print(json.dumps(payload["results"], indent=2))


if __name__ == "__main__":
    main()
