"""Analyze empirical submodularity labels."""
from __future__ import annotations

import argparse
import json

from cser.labels import CSEROracleLabels
from cser.metrics import submodularity_violation_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_npz")
    args = parser.parse_args()
    labels = CSEROracleLabels.load(args.labels_npz)
    report = {
        "n_queries": labels.n_queries,
        "n_subsets": labels.n_subsets,
        "n_experts": labels.n_experts,
        "violation_rate": submodularity_violation_rate(
            labels.marginal_values, labels.subset_masks
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
