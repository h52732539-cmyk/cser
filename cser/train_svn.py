"""Training loop for the CSER Submodular Value Network."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .labels import CSEROracleLabels
from .svn_model import (
    SubmodularValueNetwork,
    masked_mse,
    non_negative_penalty,
    submodularity_penalty,
)


@dataclass
class SVNTrainConfig:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 128
    val_fraction: float = 0.15
    lambda_submod: float = 0.1
    lambda_nonneg: float = 0.01
    patience: int = 15
    seed: int = 42
    device: str = "cpu"
    hidden: int = 128
    d_expert: int = 64
    dropout: float = 0.1


def make_training_arrays(
    query_features: np.ndarray,
    labels: CSEROracleLabels,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = []
    mask_rows = []
    target_rows = []
    for qi in range(labels.n_queries):
        for si in range(labels.n_subsets):
            target = labels.marginal_values[qi, si]
            if not np.isfinite(target).any():
                continue
            x_rows.append(query_features[qi])
            mask_rows.append(labels.subset_masks[si].astype(np.float32))
            target_rows.append(target.astype(np.float32))
    if not x_rows:
        raise ValueError("No finite marginal-value labels available for SVN training")
    return (
        np.stack(x_rows).astype(np.float32),
        np.stack(mask_rows).astype(np.float32),
        np.stack(target_rows).astype(np.float32),
    )


def _make_loaders(
    features: np.ndarray,
    masks: np.ndarray,
    targets: np.ndarray,
    config: SVNTrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    rng = np.random.default_rng(config.seed)
    n = features.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * config.val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.size == 0:
        train_idx = val_idx

    def build(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(features[idx]).float(),
            torch.from_numpy(masks[idx]).float(),
            torch.from_numpy(targets[idx]).float(),
        )
        return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle)

    return build(train_idx, True), build(val_idx, False)


def _submod_batch_loss(
    model: SubmodularValueNetwork,
    x: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    selected_optional = masks.clone()
    selected_optional[:, 0] = 0.0
    has_selected = selected_optional.sum(dim=1) > 0
    if not bool(has_selected.any()):
        return x.sum() * 0.0

    larger = masks[has_selected]
    smaller = larger.clone()
    for row in range(smaller.shape[0]):
        selected = torch.nonzero(smaller[row] > 0.5, as_tuple=False).flatten()
        selected = selected[selected != 0]
        if selected.numel() > 0:
            smaller[row, int(selected[0].item())] = 0.0

    x_sub = x[has_selected]
    pred_small = model(x_sub, smaller)
    pred_large = model(x_sub, larger)
    candidate_mask = larger < 0.5
    return submodularity_penalty(pred_small, pred_large, candidate_mask)


def train_svn(
    query_features: np.ndarray,
    labels: CSEROracleLabels,
    config: Optional[SVNTrainConfig] = None,
    save_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> Tuple[SubmodularValueNetwork, Dict[str, object]]:
    config = config or SVNTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    x_arr, mask_arr, y_arr = make_training_arrays(query_features, labels)
    train_loader, val_loader = _make_loaders(x_arr, mask_arr, y_arr, config)

    device = torch.device(config.device)
    model = SubmodularValueNetwork(
        query_dim=query_features.shape[1],
        n_experts=labels.n_experts,
        d_expert=config.d_expert,
        hidden=config.hidden,
        dropout=config.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_state = None
    best_val = float("inf")
    patience = 0
    history: Dict[str, object] = {
        "train_loss": [],
        "val_loss": [],
        "n_examples": int(x_arr.shape[0]),
        "param_count": int(model.param_count()),
    }

    t0 = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for x, masks, targets in train_loader:
            x = x.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            pred = model(x, masks)
            finite_mask = torch.isfinite(targets) & (masks < 0.5)
            loss_main = masked_mse(pred, targets)
            loss_sub = _submod_batch_loss(model, x, masks)
            loss_nonneg = non_negative_penalty(pred, finite_mask)
            loss = (
                loss_main
                + config.lambda_submod * loss_sub
                + config.lambda_nonneg * loss_nonneg
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, masks, targets in val_loader:
                x = x.to(device)
                masks = masks.to(device)
                targets = targets.to(device)
                pred = model(x, masks)
                loss = masked_mse(pred, targets)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history["train_loss"].append(train_loss)  # type: ignore[index]
        history["val_loss"].append(val_loss)  # type: ignore[index]

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if verbose and (epoch + 1) % 20 == 0:
            print(f"[svn] epoch={epoch+1} train={train_loss:.4f} val={val_loss:.4f}")
        if patience >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    history["best_val_loss"] = float(best_val)
    history["train_seconds"] = float(time.perf_counter() - t0)
    history["epochs_run"] = len(history["train_loss"])  # type: ignore[arg-type]

    if save_dir is not None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p / "best.pt")
        (p / "config.json").write_text(
            json.dumps(
                {
                    "query_dim": int(query_features.shape[1]),
                    "n_experts": int(labels.n_experts),
                    "expert_ids": list(labels.expert_ids),
                    "param_count": int(model.param_count()),
                    "history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return model, history
