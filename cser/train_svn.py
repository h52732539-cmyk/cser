"""SVN training loop (plan §4.2: MSE on marginal values + submodularity reg).

Training example layout
-----------------------
Each query contributes one row per *subset* S (16 rows). For row (q, S):

  * input  : query_feat[q] (527-D) + selected_mask(S) (K0-D)
  * target : marginal[q, S, :] (K0-D), with NaN for experts already in S

The MSE loss is masked so experts already in S do not contribute. The
submodularity regulariser penalises predicted marginals that *increase* when
the conditioning set grows (a diminishing-returns violation):

    L_sub = mean_{S ⊂ S', e∉S'} max(0, v̂(e|S') - v̂(e|S))

evaluated on the immediate-superset pairs (S, S∪{e'}) that the lattice provides.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .experts import N_OPTIONAL, all_optional_masks
from .svn import SubmodularValueNetwork
from .value_oracle import OracleLabels


@dataclass
class SVNTrainConfig:
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 256
    weight_decay: float = 1e-5
    lambda_sub: float = 0.5         # submodularity regulariser weight
    patience: int = 30
    val_fraction: float = 0.15
    d_model: int = 128
    variant: str = "full"
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_dict(cls, d: Dict) -> "SVNTrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


# ----------------------------------------------------------------------
#  Build the (query × subset) training tensors
# ----------------------------------------------------------------------

def _build_examples(labels: OracleLabels
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand oracle labels into per-(query, subset) rows.

    Returns (X_feat, X_mask, Y_marg, Y_valid):
      X_feat (M, Dq)   query feature, repeated per subset
      X_mask (M, K0)   selected-set mask
      Y_marg (M, K0)   target marginals (0 where invalid)
      Y_valid (M, K0)  1.0 where the target is a real marginal (e ∉ S)
    """
    masks = all_optional_masks().astype(np.float32)     # (2**K0, K0)
    Nq = labels.n_queries
    n_subsets = labels.n_subsets

    X_feat = np.repeat(labels.query_feats, n_subsets, axis=0)
    X_mask = np.tile(masks, (Nq, 1))
    Y = labels.marginal.reshape(Nq * n_subsets, N_OPTIONAL)
    Y_valid = (~np.isnan(Y)).astype(np.float32)
    Y_marg = np.nan_to_num(Y, nan=0.0).astype(np.float32)
    return X_feat.astype(np.float32), X_mask, Y_marg, Y_valid


def _submodularity_penalty(model: SubmodularValueNetwork,
                           X_feat: torch.Tensor) -> torch.Tensor:
    """Diminishing-returns penalty over immediate-superset pairs.

    For a batch of queries, compare predicted marginals at the empty set vs.
    each singleton set. v̂(e | {e'}) must not exceed v̂(e | ∅) for e ≠ e'.
    Cheap O(K0) probe that captures the core submodular constraint.
    """
    B = X_feat.shape[0]
    K0 = model.n_experts
    empty = torch.zeros(B, K0, device=X_feat.device)
    base = model(X_feat, empty)                          # v̂(e | ∅)  (B, K0)

    penalties = []
    for ep in range(K0):                                 # add singleton {ep}
        m = torch.zeros(B, K0, device=X_feat.device)
        m[:, ep] = 1.0
        v = model(X_feat, m)                             # v̂(e | {ep})
        diff = v - base                                  # want ≤ 0 for e ≠ ep
        diff = diff.clone()
        diff[:, ep] = 0.0                                # ignore the added expert
        penalties.append(torch.clamp(diff, min=0.0))
    return torch.stack(penalties, dim=0).mean()


# ----------------------------------------------------------------------
#  Train
# ----------------------------------------------------------------------

def train_svn(labels: OracleLabels,
              config: Optional[SVNTrainConfig] = None,
              save_dir: Optional[str] = None,
              verbose: bool = True) -> Tuple[SubmodularValueNetwork, Dict]:
    config = config or SVNTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    X_feat, X_mask, Y_marg, Y_valid = _build_examples(labels)
    M = X_feat.shape[0]

    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(M)
    n_val = max(1, int(M * config.val_fraction))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def _loader(idx, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X_feat[idx]), torch.from_numpy(X_mask[idx]),
            torch.from_numpy(Y_marg[idx]), torch.from_numpy(Y_valid[idx]),
        )
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            pin_memory=device.type == "cuda",
        )

    train_loader = _loader(tr_idx, True)
    val_loader = _loader(val_idx, False)

    model = SubmodularValueNetwork(
        d_query=X_feat.shape[1], d_model=config.d_model,
        n_experts=N_OPTIONAL, variant=config.variant,
    ).to(device)
    if verbose:
        print(f"[svn] variant={config.variant} params={model.param_count():,} "
              f"device={device} batch_size={config.batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)

    def _masked_mse(pred, target, valid):
        se = (pred - target) ** 2 * valid
        return se.sum() / valid.sum().clamp(min=1.0)

    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"train_loss": [], "val_mse": []}

    t0 = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        tl = []
        for Xf, Xm, Ym, Yv in train_loader:
            Xf = Xf.to(device, non_blocking=True)
            Xm = Xm.to(device, non_blocking=True)
            Ym = Ym.to(device, non_blocking=True)
            Yv = Yv.to(device, non_blocking=True)
            pred = model(Xf, Xm)
            loss = _masked_mse(pred, Ym, Yv)
            if config.lambda_sub > 0:
                loss = loss + config.lambda_sub * _submodularity_penalty(model, Xf)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for Xf, Xm, Ym, Yv in val_loader:
                Xf = Xf.to(device, non_blocking=True)
                Xm = Xm.to(device, non_blocking=True)
                Ym = Ym.to(device, non_blocking=True)
                Yv = Yv.to(device, non_blocking=True)
                vl.append(_masked_mse(model(Xf, Xm), Ym, Yv).item())

        tr_avg, val_avg = float(np.mean(tl)), float(np.mean(vl))
        history["train_loss"].append(tr_avg)
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
        print(f"[svn] done in {elapsed:.1f}s best_val_mse={best_val:.5f}")

    if save_dir:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p / "svn.pt")
        (p / "svn_config.json").write_text(json.dumps({
            "variant": config.variant,
            "d_query": int(X_feat.shape[1]),
            "d_model": config.d_model,
            "param_count": model.param_count(),
            "best_val_mse": best_val,
            "epochs_run": len(history["train_loss"]),
            "train_seconds": elapsed,
            "lambda_sub": config.lambda_sub,
            "epochs_requested": config.epochs,
            "seed": config.seed,
            "patience": config.patience,
            "training_device": str(device),
            "batch_size": config.batch_size,
        }, indent=2), encoding="utf-8")
        if verbose:
            print(f"[save] {p / 'svn.pt'}")

    return model, history
