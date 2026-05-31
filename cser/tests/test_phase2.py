"""Phase-2 tests: conformal gate coverage, pipeline, baselines (real-expert API)."""
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
from cser.conformal import (SplitConformal, MondrianConformal, conformal_quantile,
                            gt_nonconformity, qpp_margin, evaluate_coverage)
from cser.pipeline import CSERPipeline
from cser.baselines import (AllExperts, RandomSelect, FixedCascade, UCBBandit,
                            oracle_mask)
from cser.experts import N_OPTIONAL, SEMANTIC_COST, OPTIONAL_COSTS, mask_to_id


@pytest.fixture(scope="module")
def setup():
    ds = build_synthetic_dataset(n_videos=60, n_queries=120, seed=1)
    eng = RetrievalEngine(ds.gallery)
    tr, cal, te = ds.split(seed=1)

    def sub(idx):
        return ([ds.query_priors[i] for i in idx], [ds.gt_video_ids[i] for i in idx])

    p, g = sub(tr); oracle_tr = build_oracle_labels(eng, p, g, verbose=False)
    p_cal, g_cal = sub(cal); oracle_cal = build_oracle_labels(eng, p_cal, g_cal, verbose=False)
    p_te, g_te = sub(te); oracle_te = build_oracle_labels(eng, p_te, g_te, verbose=False)
    model, _ = train_svn(oracle_tr, SVNTrainConfig(epochs=30, patience=30, seed=1),
                         verbose=False)
    fdim = oracle_tr.feature_dim
    return dict(ds=ds, eng=eng, model=model, oracle_te=oracle_te,
                p_cal=p_cal, g_cal=g_cal, p_te=p_te, g_te=g_te, fdim=fdim)


# ---- conformal math ----

def test_conformal_quantile_small_n_inf():
    assert conformal_quantile(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 0.01) == float("inf")


def test_conformal_quantile_monotone():
    s = np.linspace(0, 1, 100)
    assert conformal_quantile(s, 0.01) >= conformal_quantile(s, 0.20)


def test_gt_nonconformity_range():
    sn = np.array([0.2, 0.9, 0.5], np.float32)
    assert gt_nonconformity(sn, 1) == pytest.approx(0.1, abs=1e-6)
    assert gt_nonconformity(sn, -1) == 1.0


def _sims_gidx(eng, priors, gts):
    return [eng.semantic_norm(p) for p in priors], [eng.id_to_idx(g) for g in gts]


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_split_conformal_coverage(setup, alpha):
    eng = setup["eng"]
    cal_sims, cal_gidx = _sims_gidx(eng, setup["p_cal"], setup["g_cal"])
    te_sims, te_gidx = _sims_gidx(eng, setup["p_te"], setup["g_te"])
    scores = np.array([gt_nonconformity(cal_sims[i], cal_gidx[i])
                       for i in range(len(cal_gidx))])
    gate = SplitConformal.calibrate(scores, alpha)
    rep = evaluate_coverage(gate, te_sims, te_gidx)
    assert rep.empirical_coverage >= (1 - alpha) - 0.12   # finite-sample slack


def test_mondrian_bins(setup):
    eng = setup["eng"]
    cal_sims, cal_gidx = _sims_gidx(eng, setup["p_cal"], setup["g_cal"])
    margins = np.array([qpp_margin(s) for s in cal_sims])
    scores = np.array([gt_nonconformity(cal_sims[i], cal_gidx[i])
                       for i in range(len(cal_gidx))])
    gate = MondrianConformal.calibrate(scores, margins, 0.10, n_bins=3)
    assert len(gate.thresholds) == 3 and len(gate.bin_edges) == 2


# ---- pipeline ----

def test_pipeline_respects_budget(setup):
    eng, model, oracle = setup["eng"], setup["model"], setup["oracle_te"]
    for B in (1.0, 5.0, 9.5):
        pipe = CSERPipeline(eng, model, conformal_gate=None, budget=B)
        res = pipe.run(setup["p_te"][0], oracle.query_feats[0], setup["g_te"][0])
        assert res.cost <= B + 1e-6
        assert not res.gt_filtered


def test_pipeline_coverage(setup):
    eng, model, oracle = setup["eng"], setup["model"], setup["oracle_te"]
    cal_sims, cal_gidx = _sims_gidx(eng, setup["p_cal"], setup["g_cal"])
    scores = np.array([gt_nonconformity(cal_sims[i], cal_gidx[i])
                       for i in range(len(cal_gidx))])
    gate = SplitConformal.calibrate(scores, 0.10)
    pipe = CSERPipeline(eng, model, conformal_gate=gate, budget=5.0)
    covered = []
    for i in range(len(setup["g_te"])):
        res = pipe.run(setup["p_te"][i], oracle.query_feats[i], setup["g_te"][i])
        covered.append(res.gt_in_conformal_set)
        assert 1 <= res.conformal_set_size <= eng._N
    assert np.mean(covered) >= 0.75


# ---- baselines ----

def test_all_experts_full(setup):
    m = AllExperts(budget=float("inf")).select(np.zeros(setup["fdim"], np.float32))
    assert m.all()


def test_random_cascade_budget(setup):
    for pol in (RandomSelect(budget=5.0, seed=0), FixedCascade(budget=5.0)):
        m = pol.select(np.zeros(setup["fdim"], np.float32))
        assert SEMANTIC_COST + OPTIONAL_COSTS[m].sum() <= 5.0 + 1e-6


def test_ucb_learns(setup):
    b = UCBBandit(budget=9.5, seed=0)
    for _ in range(40):
        b.select(np.zeros(setup["fdim"], np.float32)); b.update(0.5)
    assert b.t == 40 and b.counts.sum() > 0


def test_oracle_mask_best(setup):
    row = setup["oracle_te"].value_matrix[0]
    m = oracle_mask(row, budget=9.5)
    assert row[mask_to_id(m)] == row.max()
