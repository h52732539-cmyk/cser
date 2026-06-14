"""Reusable CSER oracle lattices and trained selector models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from .experts import N_OPTIONAL, OPTIONAL_NAMES
from .set_value_network import SetValueNetwork
from .svn import SubmodularValueNetwork
from .train_set_value import SetValueTrainConfig, train_set_value
from .train_svn import SVNTrainConfig, train_svn
from .value_oracle import OracleLabels, build_oracle_labels, query_feature_dim


def _update_strings(digest, values: Sequence[str]) -> None:
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")


def dataset_fingerprint(dataset, split_indices, metric: str, seed: int) -> str:
    """Hash the data ordering and split used by reusable training artifacts."""
    digest = hashlib.sha256()
    digest.update(f"metric={metric}\nseed={seed}\n".encode("ascii"))
    _update_strings(digest, dataset.video_ids)
    _update_strings(digest, dataset.query_texts)
    _update_strings(digest, dataset.gt_video_ids)
    for name, idx in zip(("train", "cal", "test"), split_indices):
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(idx, dtype=np.int64).tobytes())
    return digest.hexdigest()


def prepare_artifact_dir(artifact_dir: str, dataset, split_indices,
                         metric: str, seed: int) -> Tuple[Path, Dict]:
    """Create or validate the manifest guarding a shared artifact directory."""
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = {
        "kind": "cser_shared_artifacts",
        "version": 1,
        "dataset_fingerprint": dataset_fingerprint(
            dataset, split_indices, metric, seed),
        "metric": metric,
        "seed": int(seed),
        "gallery_size": int(dataset.gallery_size),
        "n_queries": int(dataset.n_queries),
        "split": {
            "train": int(len(split_indices[0])),
            "cal": int(len(split_indices[1])),
            "test": int(len(split_indices[2])),
        },
        "optional_experts": list(OPTIONAL_NAMES),
        "gallery_cache_manifest": dataset.cache_manifest,
    }
    path = root / "artifact_manifest.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        for key in ("dataset_fingerprint", "metric", "seed", "gallery_size",
                    "n_queries", "split", "optional_experts"):
            if current.get(key) != expected[key]:
                raise RuntimeError(
                    f"shared artifact manifest mismatch for {key}: "
                    f"{current.get(key)!r} != {expected[key]!r}")
        return root, current
    path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return root, expected


def load_or_build_oracle(root: Path, split_name: str, engine, priors,
                         gt_video_ids, metric: str) -> Tuple[OracleLabels, bool]:
    """Load one persisted oracle lattice, or build and save it once."""
    path = root / f"oracle_{split_name}.npz"
    if path.exists():
        labels = OracleLabels.load(str(path))
        expected_dim = query_feature_dim(engine.g.clip_dim)
        if labels.metric != metric:
            raise RuntimeError(
                f"{path} metric {labels.metric!r} does not match {metric!r}")
        if labels.n_queries != len(gt_video_ids):
            raise RuntimeError(
                f"{path} has {labels.n_queries} queries, expected "
                f"{len(gt_video_ids)}")
        if labels.feature_dim != expected_dim:
            raise RuntimeError(
                f"{path} feature dim {labels.feature_dim}, expected "
                f"{expected_dim}")
        if tuple(labels.optional_experts) != tuple(OPTIONAL_NAMES):
            raise RuntimeError(f"{path} expert roster does not match the code")
        print(f"[artifact] loaded oracle {split_name}: {path}")
        return labels, True

    print(f"[artifact] building oracle {split_name}: {path}")
    labels = build_oracle_labels(
        engine, priors, gt_video_ids, metric=metric, verbose=False)
    labels.save(str(path))
    return labels, False


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_svn_model(model_path: Path, labels: OracleLabels,
                   device: str = "cpu") -> Tuple[SubmodularValueNetwork, Dict]:
    config_path = model_path.with_name("svn_config.json")
    if not config_path.exists():
        raise RuntimeError(f"missing SVN config next to model: {config_path}")
    meta = _read_json(config_path)
    if int(meta["d_query"]) != labels.feature_dim:
        raise RuntimeError(
            f"{model_path} d_query={meta['d_query']} does not match "
            f"{labels.feature_dim}")
    model = SubmodularValueNetwork(
        d_query=labels.feature_dim,
        d_model=int(meta.get("d_model", 128)),
        n_experts=N_OPTIONAL,
        variant=str(meta.get("variant", "full")),
    )
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    return model.to(device).eval(), meta


def load_or_train_svn(labels: OracleLabels, config: SVNTrainConfig,
                      model_dir: Path, verbose: bool = True
                      ) -> Tuple[SubmodularValueNetwork, Dict, bool]:
    """Load a compatible SVN checkpoint or train it into ``model_dir``."""
    model_path = model_dir / "svn.pt"
    config_path = model_dir / "svn_config.json"
    if model_path.exists() and config_path.exists():
        model, meta = load_svn_model(model_path, labels)
        expected = {
            "variant": config.variant,
            "d_model": int(config.d_model),
            "lambda_sub": float(config.lambda_sub),
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(
                    f"{config_path} mismatch for {key}: "
                    f"{meta.get(key)!r} != {value!r}")
        print(f"[artifact] loaded SVN model: {model_path}")
        return model, meta, True

    model, history = train_svn(
        labels, config, save_dir=str(model_dir), verbose=verbose)
    meta = _read_json(config_path)
    meta["history"] = history
    return model, meta, False


def load_set_value_model(model_path: Path, labels: OracleLabels,
                         device: str = "cpu") -> Tuple[SetValueNetwork, Dict]:
    config_path = model_path.with_name("set_value_config.json")
    if not config_path.exists():
        raise RuntimeError(
            f"missing set-value config next to model: {config_path}")
    meta = _read_json(config_path)
    if int(meta["d_query"]) != labels.feature_dim:
        raise RuntimeError(
            f"{model_path} d_query={meta['d_query']} does not match "
            f"{labels.feature_dim}")
    model = SetValueNetwork(
        d_query=labels.feature_dim,
        d_model=int(meta.get("d_model", 128)),
        n_experts=N_OPTIONAL,
    )
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    return model.to(device).eval(), meta


def load_or_train_set_value(labels: OracleLabels,
                            config: SetValueTrainConfig,
                            model_dir: Path,
                            verbose: bool = True
                            ) -> Tuple[SetValueNetwork, Dict, bool]:
    """Load a compatible set-value model or train it once."""
    model_path = model_dir / "set_value.pt"
    config_path = model_dir / "set_value_config.json"
    if model_path.exists() and config_path.exists():
        model, meta = load_set_value_model(model_path, labels)
        if int(meta.get("d_model", 128)) != int(config.d_model):
            raise RuntimeError(
                f"{config_path} d_model does not match requested config")
        print(f"[artifact] loaded set-value model: {model_path}")
        return model, meta, True

    model, history = train_set_value(
        labels, config, save_dir=str(model_dir), verbose=verbose)
    meta = _read_json(config_path)
    meta["history"] = history
    return model, meta, False
