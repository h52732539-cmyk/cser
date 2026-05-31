"""Phase-3 tests: E7-E10 experiments + empirical theorem verification (real-expert API)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cser.data import build_synthetic_dataset
from cser.retrieval import RetrievalEngine
from cser.value_oracle import build_oracle_labels
from cser.train_svn import train_svn, SVNTrainConfig
from cser.conformal import MondrianConformal, gt_nonconformity, qpp_margin
from cser.submodularity import verify_submodularity
from cser.experiments_extra import (exp_e7_scalability, exp_e8_robustness,
                                    exp_e9_expert_contribution,
                                    exp_e10_oracle_comparison)
from cser.theory import (verify_theorem1_coverage, verify_theorem2_greedy,
                         verify_theorem3_combined, measure_surrogate_error)
from cser.experts import N_OPTIONAL, OPTIONAL_COSTS, SEMANTIC_COST


@pytest.fixture(scope="module")
def setup():
    ds = build_synthetic_dataset(n_videos=120, n_queries=120, seed=2)
    eng = RetrievalEngine(ds.gallery)
    tr, cal, te = ds.split(seed=2)

    def sub(idx):
        return ([ds.query_priors[i] for i in idx], [ds.gt_video_ids[i] for i in idx])

    p, g = sub(tr); oracle_tr = build_oracle_labels(eng, p, g, verbose=False)
    p_te, g_te = sub(te); oracle_te = build_oracle_labels(eng, p_te, g_te, verbose=False)
    p_cal, g_cal = sub(cal)
    model, _ = train_svn(oracle_tr, SVNTrainConfig(epochs=40, patience=40, seed=2),
                         verbose=False)
    return dict(ds=ds, eng=eng, model=model, oracle_te=oracle_te,
                p_te=p_te, g_te=g_te, p_cal=p_cal, g_cal=g_cal)


def test_e7_scalability(setup):
    e7 = exp_e7_scalability(setup["ds"], setup["model"], budget=5.0,
                            sizes=(40, 80), seed=2)
    assert len(e7) == 2
    for v in e7.values():
        assert v["speedup"] > 0 and 0.0 <= v["cser_R@1"] <= 1.0


def test_e8_robustness(setup):
    e8 = exp_e8_robustness(setup["ds"], setup["model"], budget=5.0, seed=2)
    assert set(e8.keys()) == {"clean", "mild", "medium", "heavy"}
    for v in e8.values():
        assert v["cser_GT_filtered"] == 0.0 and 0.0 <= v["cser_R@1"] <= 1.0


def test_e9_expert_contribution(setup):
    e9 = exp_e9_expert_contribution(setup["oracle_te"], setup["model"])
    assert len(e9["per_expert"]) == N_OPTIONAL
    assert len(e9["expert_ranking_by_value"]) == N_OPTIONAL
    assert -1.0 <= e9["svn_oracle_marginal_correlation"] <= 1.0


def test_e10_oracle_comparison(setup):
    e10 = exp_e10_oracle_comparison(setup["oracle_te"], setup["model"], budget=5.0)
    assert e10["oracle_best_subset"]["pct_of_oracle"] == pytest.approx(1.0, abs=1e-6)
    for v in e10.values():
        assert v["pct_of_oracle"] <= 1.0 + 1e-6
    assert (e10["greedy_true_values"]["mean_value"]
            >= e10["semantic_only"]["mean_value"] - 1e-6)


def test_surrogate_error_nonneg(setup):
    assert measure_surrogate_error(setup["model"], setup["oracle_te"]) >= 0.0


def test_theorem1_coverage(setup):
    eng = setup["eng"]
    cal_sim = [eng.semantic_norm(p) for p in setup["p_cal"]]
    cal_gidx = [eng.id_to_idx(g) for g in setup["g_cal"]]
    margins = np.array([qpp_margin(s) for s in cal_sim])
    scores = np.array([gt_nonconformity(cal_sim[i], cal_gidx[i])
                       for i in range(len(cal_gidx))])
    gate = MondrianConformal.calibrate(scores, margins, 0.10, 3)
    te_sim = [eng.semantic_norm(p) for p in setup["p_te"]]
    te_gidx = [eng.id_to_idx(g) for g in setup["g_te"]]
    thm1 = verify_theorem1_coverage(gate, te_sim, te_gidx)
    assert isinstance(thm1["holds"], bool)
    assert 0.0 <= thm1["empirical_coverage"] <= 1.0


def test_theorem2_bound(setup):
    submod = verify_submodularity(setup["oracle_te"])
    thm2 = verify_theorem2_greedy(setup["model"], setup["oracle_te"],
                                  submod.gamma_ratio_p10, budget=5.0)
    assert thm2["realised_value_LHS"] >= thm2["bound_RHS"] - 1e-6
    assert 0.0 < thm2["approx_factor_(1-e^-gamma)"] <= 1.0


def test_theorem3_combined(setup):
    submod = verify_submodularity(setup["oracle_te"])
    thm2 = verify_theorem2_greedy(setup["model"], setup["oracle_te"],
                                  submod.gamma_ratio_p10, budget=5.0)
    thm3 = verify_theorem3_combined({"holds": True}, thm2,
                                    float(SEMANTIC_COST + OPTIONAL_COSTS[0]), 5.0)
    assert thm3["budget_compliance_holds"]
    assert isinstance(thm3["all_three_hold"], bool)
