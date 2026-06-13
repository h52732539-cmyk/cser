"""Phase-1 tests for the CSER package (real-expert / mock-model API).

Run:  cd litevtr_multi_model_framework && python -m pytest cser/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cser import experts
from cser.data import build_synthetic_dataset
from cser.expert_features import build_model_bundle, build_query_priors
from cser.retrieval import RetrievalEngine
from cser.value_oracle import build_oracle_labels, OracleLabels, query_feature_dim
from cser.train_svn import train_svn, SVNTrainConfig, _build_examples
from cser.svn import SubmodularValueNetwork, VARIANTS
from cser.set_value_network import SetValueNetwork
from cser.train_set_value import train_set_value, SetValueTrainConfig
from cser.greedy import GreedyBudgetedSelector
from cser.submodularity import verify_submodularity


# ----------------------------------------------------------------------
#  experts.py
# ----------------------------------------------------------------------

def test_expert_roster():
    assert experts.N_EXPERTS == 5
    assert experts.N_OPTIONAL == 4
    assert experts.EXPERTS[experts.SEMANTIC_IDX].mandatory
    assert experts.OPTIONAL_NAMES == ("highlight", "face", "face_id", "scene")


def test_mask_id_roundtrip():
    masks = experts.all_optional_masks()
    assert masks.shape == (16, 4)
    for m in masks:
        assert np.array_equal(experts.id_to_mask(experts.mask_to_id(m)), m)


def test_selection_cost():
    empty = np.zeros(experts.N_OPTIONAL, bool)
    full = np.ones(experts.N_OPTIONAL, bool)
    assert experts.selection_cost(empty) == experts.SEMANTIC_COST
    assert experts.selection_cost(full) == pytest.approx(
        experts.SEMANTIC_COST + experts.OPTIONAL_COSTS.sum())


# ----------------------------------------------------------------------
#  model bundle + data
# ----------------------------------------------------------------------

def test_mock_bundle_builds():
    b = build_model_bundle(use_real=False)
    embs = b.clip.encode_text(["a person at the beach"])
    assert len(embs) == 1 and embs[0].ndim == 1


def test_real_bundle_init_failure_does_not_fallback(monkeypatch):
    from tasks import real_models

    def _missing_weights():
        raise RuntimeError("missing test checkpoint")

    monkeypatch.setattr(real_models, "RealCLIPModel", _missing_weights)
    with pytest.raises(RuntimeError, match="CSER real-model init failed"):
        build_model_bundle(use_real=True)


@pytest.fixture(scope="module")
def dataset():
    return build_synthetic_dataset(n_videos=40, n_queries=40, seed=0)


def test_synthetic_dataset_shapes(dataset):
    assert dataset.gallery_size == 40
    assert dataset.n_queries == 40
    assert len(dataset.query_priors) == 40


def test_gallery_signals_present(dataset):
    g = dataset.gallery
    assert g.clip_matrix().shape[0] == 40
    assert g.highlight_vector().shape == (40,)
    assert g.face_vector().shape == (40,)


# ----------------------------------------------------------------------
#  retrieval value function
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine(dataset):
    return RetrievalEngine(dataset.gallery)


def test_value_in_unit_interval(dataset, engine):
    v = engine.value(dataset.query_priors[0], dataset.gt_video_ids[0],
                     active_experts=[], metric="rr")
    assert 0.0 <= v <= 1.0


def test_selecting_experts_changes_scores(dataset, engine):
    p = dataset.query_priors[0]
    base = engine.final_scores(p, [])
    with_all = engine.final_scores(p, list(experts.OPTIONAL_NAMES))
    assert not np.allclose(base, with_all)   # experts must move the scores


# ----------------------------------------------------------------------
#  oracle
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def oracle(dataset, engine):
    return build_oracle_labels(engine, dataset.query_priors,
                               dataset.gt_video_ids, metric="rr", verbose=False)


def test_oracle_shapes(oracle, dataset):
    assert oracle.value_matrix.shape == (dataset.n_queries, 16)
    assert oracle.marginal.shape == (dataset.n_queries, 16, 4)
    assert oracle.query_feats.shape[1] == query_feature_dim(dataset.gallery.clip_dim)


def test_marginal_consistency(oracle):
    V = oracle.value_matrix
    for sid in range(16):
        for j in range(4):
            if (sid >> j) & 1:
                assert np.isnan(oracle.marginal[:, sid, j]).all()
            else:
                np.testing.assert_allclose(
                    oracle.marginal[:, sid, j],
                    V[:, sid | (1 << j)] - V[:, sid], atol=1e-5)


def test_oracle_save_load(oracle, tmp_path):
    p = tmp_path / "o.npz"
    oracle.save(str(p))
    loaded = OracleLabels.load(str(p))
    np.testing.assert_allclose(loaded.value_matrix, oracle.value_matrix)


# ----------------------------------------------------------------------
#  SVN + training + greedy + submodularity
# ----------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_svn_forward_shapes(variant, oracle):
    d = oracle.feature_dim
    model = SubmodularValueNetwork(d_query=d, variant=variant)
    import torch
    out = model(torch.zeros(8, d), torch.zeros(8, 4))
    assert out.shape == (8, 4)


def test_set_value_forward_shapes(oracle):
    d = oracle.feature_dim
    model = SetValueNetwork(d_query=d)
    import torch
    out = model(torch.zeros(8, d))
    assert out.shape == (8, 16)


def test_build_examples_shapes(oracle):
    Xf, Xm, Ym, Yv = _build_examples(oracle)
    M = oracle.n_queries * 16
    assert Xf.shape == (M, oracle.feature_dim)
    assert Xm.shape == (M, 4) and Ym.shape == (M, 4)
    assert 0 < Yv.sum() < M * 4


def test_svn_trains(oracle):
    model, history = train_svn(oracle, SVNTrainConfig(epochs=5, patience=5, seed=0),
                               verbose=False)
    assert min(history["val_mse"]) <= history["val_mse"][0] + 1e-6


def test_set_value_trains(oracle):
    model, history = train_set_value(
        oracle, SetValueTrainConfig(epochs=3, patience=3, seed=0),
        verbose=False)
    assert model.n_subsets == 16
    assert len(history["val_mse"]) >= 1
    assert np.isfinite(history["val_mse"][-1])


def test_greedy_respects_budget(oracle):
    model, _ = train_svn(oracle, SVNTrainConfig(epochs=3, patience=3, seed=0),
                         verbose=False)
    for B in (1.0, 3.0, 5.0, 9.5):
        r = GreedyBudgetedSelector(model, budget=B).select(oracle.query_feats[0])
        assert r.cost <= B + 1e-6
        assert r.n_experts_called == 1 + int(r.selected_mask.sum())


def test_greedy_budget1_only_semantic(oracle):
    model, _ = train_svn(oracle, SVNTrainConfig(epochs=1, patience=1, seed=0),
                         verbose=False)
    r = GreedyBudgetedSelector(model, budget=experts.SEMANTIC_COST).select(
        oracle.query_feats[0])
    assert r.selected_mask.sum() == 0


def test_submodularity_report_wellformed(oracle):
    rep = verify_submodularity(oracle)
    assert 0.0 <= rep.submodularity_violation_rate <= 1.0
    assert 0.0 <= rep.gamma_ratio_mean <= 1.0
    assert rep.verdict in ("submodular", "weakly_submodular", "non_submodular")
