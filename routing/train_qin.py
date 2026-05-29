"""C-QIN training loop.

Loss = loss_value + α·loss_route_ce + β·loss_safety

  loss_value:     Huber regression on normalized route utilities
  loss_route_ce:  cross-entropy on oracle route id
  loss_safety:    BCE on per-axis GT survival labels
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .qin_model import CalibratedQIN
from .route_bank_builder import RouteBankLabels


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 128
    alpha_ce: float = 1.0
    beta_safety: float = 1.0
    patience: int = 20
    val_fraction: float = 0.15
    seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ----------------------------------------------------------------------

def _normalize_utilities(utils: np.ndarray) -> np.ndarray:
    """Per-query normalization to [0, 1]."""
    u_min = utils.min(axis=1, keepdims=True)
    u_max = utils.max(axis=1, keepdims=True)
    denom = u_max - u_min
    denom[denom < 1e-8] = 1.0
    return (utils - u_min) / denom


def prepare_data(
    features: np.ndarray,
    labels: RouteBankLabels,
    config: TrainConfig,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """Split into train / val and return DataLoaders.

    Returns (train_loader, val_loader, val_indices).
    """
    Nq = features.shape[0]
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(Nq)
    n_val = max(1, int(Nq * config.val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    norm_utils = _normalize_utilities(labels.utilities)

    def _make_loader(idx):
        X = torch.from_numpy(features[idx]).float()
        Y_util = torch.from_numpy(norm_utils[idx]).float()
        Y_oracle = torch.from_numpy(labels.oracle_route_idx[idx]).long()
        Y_safety = torch.from_numpy(
            labels.survival_labels[idx].astype(np.float32)
        ).float()
        ds = TensorDataset(X, Y_util, Y_oracle, Y_safety)
        return DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    return _make_loader(train_idx), _make_loader(val_idx), val_idx


# ----------------------------------------------------------------------

def train_cqin(
    features: np.ndarray,
    labels: RouteBankLabels,
    config: Optional[TrainConfig] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[CalibratedQIN, Dict]:
    """Train C-QIN from counterfactual route labels."""
    config = config or TrainConfig()
    input_dim = features.shape[1]
    num_routes = labels.n_routes
    device = torch.device(config.device)

    model = CalibratedQIN(
        input_dim=input_dim,
        num_routes=num_routes,
        num_safety_axes=4,
    ).to(device)

    if verbose:
        print(f"[train] C-QIN params: {model.param_count():,}")
        assert model.param_count() < 100_000, \
            f"Model too large: {model.param_count()} > 100K"

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    huber = nn.HuberLoss()
    bce = nn.BCEWithLogitsLoss()

    train_loader, val_loader, val_idx = prepare_data(features, labels, config)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    t0 = time.perf_counter()
    for epoch in range(config.epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for X, Y_util, Y_oracle, Y_safety in train_loader:
            X = X.to(device)
            Y_util = Y_util.to(device)
            Y_oracle = Y_oracle.to(device)
            Y_safety = Y_safety.to(device)

            out = model(X)
            loss_val = huber(out["route_values"], Y_util)
            loss_ce = F.cross_entropy(out["route_values"], Y_oracle)
            loss_safe = bce(out["safety_logits"], Y_safety)
            loss = loss_val + config.alpha_ce * loss_ce + config.beta_safety * loss_safe

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, Y_util, Y_oracle, Y_safety in val_loader:
                X = X.to(device)
                Y_util = Y_util.to(device)
                Y_oracle = Y_oracle.to(device)
                Y_safety = Y_safety.to(device)
                out = model(X)
                loss_val = huber(out["route_values"], Y_util)
                loss_ce = F.cross_entropy(out["route_values"], Y_oracle)
                loss_safe = bce(out["safety_logits"], Y_safety)
                loss = loss_val + config.alpha_ce * loss_ce + config.beta_safety * loss_safe
                val_losses.append(loss.item())

        train_avg = float(np.mean(train_losses))
        val_avg = float(np.mean(val_losses))
        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_avg)

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1:3d}  train={train_avg:.4f}  "
                  f"val={val_avg:.4f}  best={best_val_loss:.4f}")

        if patience_counter >= config.patience:
            if verbose:
                print(f"  early stopping at epoch {epoch+1}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"[train] done in {elapsed:.1f}s  "
              f"best_val_loss={best_val_loss:.4f}")

    # Save
    if save_dir:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p / "best.pt")
        (p / "config.json").write_text(
            json.dumps({
                "input_dim": input_dim,
                "num_routes": num_routes,
                "param_count": model.param_count(),
                "epochs_run": len(history["train_loss"]),
                "best_val_loss": best_val_loss,
                "train_seconds": elapsed,
            }, indent=2),
            encoding="utf-8",
        )
        if verbose:
            print(f"[save] {p / 'best.pt'}")

    return model, history
