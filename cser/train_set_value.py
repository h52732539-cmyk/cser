"""Training loop for :class:`cser.set_value_network.SetValueNetwork`."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .experts import N_OPTIONAL
from .set_value_network import SetValueNetwork
from .value_oracle import OracleLabels


@dataclass
class SetValueTrainConfig:
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 128
    weight_decay: float = 1e-5
    patience: int = 30
    val_fraction: float = 0.15
    d_model: int = 128
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_dict(cls, d: Dict) -> "SetValueTrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def train_set_value(labels: OracleLabels,
                    config: Optional[SetValueTrainConfig] = None,
                    save_dir: Optional[str] = None,
                    verbose: bool = True) -> Tuple[SetValueNetwork, Dict]:
    """Train a set-value predictor from ``OracleLabels.value_matrix``."""
    config = config or SetValueTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    X = labels.query_feats.astype(np.float32)
    Y = labels.value_matrix.astype(np.float32)
    n = X.shape[0]

    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * config.val_fraction)) if n > 1 else 0
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if len(tr_idx) == 0:
        tr_idx, val_idx = perm, perm[:0]

    def _loader(idx, shuffle):
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(Y[idx]))
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            pin_memory=device.type == "cuda",
        )

    train_loader = _loader(tr_idx, True)
    val_loader = _loader(val_idx, False) if len(val_idx) else None

    model = SetValueNetwork(
        d_query=X.shape[1], d_model=config.d_model, n_experts=N_OPTIONAL,
    ).to(device)
    if verbose:
        print(f"[set-value] params={model.param_count():,} "
              f"device={device} batch_size={config.batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"train_mse": [], "val_mse": []}

    t0 = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        tl = []
        for Xb, Yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            Yb = Yb.to(device, non_blocking=True)
            loss = loss_fn(model(Xb), Yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(float(loss.item()))

        model.eval()
        if val_loader is None:
            val_avg = float(np.mean(tl))
        else:
            vl = []
            with torch.no_grad():
                for Xb, Yb in val_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    Yb = Yb.to(device, non_blocking=True)
                    vl.append(float(loss_fn(model(Xb), Yb).item()))
            val_avg = float(np.mean(vl))

        tr_avg = float(np.mean(tl))
        history["train_mse"].append(tr_avg)
        history["val_mse"].append(val_avg)

        if val_avg < best_val:
            best_val = val_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch+1:3d} train={tr_avg:.5f} "
                  f"val_mse={val_avg:.5f} best={best_val:.5f}")
        if patience >= config.patience:
            if verbose:
                print(f"  early stop @ epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.to("cpu")
    model.eval()
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"[set-value] done in {elapsed:.1f}s best_val_mse={best_val:.5f}")

    if save_dir:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p / "set_value.pt")
        (p / "set_value_config.json").write_text(json.dumps({
            "d_query": int(X.shape[1]),
            "d_model": config.d_model,
            "n_subsets": int(Y.shape[1]),
            "param_count": model.param_count(),
            "best_val_mse": best_val,
            "epochs_run": len(history["train_mse"]),
            "train_seconds": elapsed,
            "training_device": str(device),
            "batch_size": config.batch_size,
        }, indent=2), encoding="utf-8")
        if verbose:
            print(f"[save] {p / 'set_value.pt'}")

    return model, history
