#!/usr/bin/env python3
"""
Train a simple ANN on prepared tensors (numeric + categorical embeddings).

Why this file exists:
  - Our categoricals (ex: PropertyStyle) have thousands of unique values.
    One-hot encoding would create a *huge* dense feature vector, which is
    not practical for a small ANN.
  - Instead, we learn an embedding for each categorical column and
    concatenate those embeddings with the numeric features.

What we predict:
  - We train on `y_log1p = log1p(TransactionPrice)` because raw prices have
    a heavy tail; log-space usually trains more smoothly.

Outputs:
  - Prints per-epoch validation loss and a final summary including test loss.
"""

from __future__ import annotations

import argparse
import copy
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class TensorDataset(Dataset):
    """Tiny wrapper around the tensors saved by prepare_ann_tensors.py."""

    def __init__(self, payload: dict):
        self.x_num = payload["X_num"]
        self.x_cat = payload["X_cat"]
        self.y = payload["y_log1p"]

    def __len__(self) -> int:
        return self.x_num.shape[0]

    def __getitem__(self, idx: int):
        # Return tensors directly so DataLoader can collate into batches.
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class PriceANN(nn.Module):
    """MLP that consumes numeric features + categorical embeddings."""

    def __init__(
        self,
        *,
        n_num: int,
        cat_sizes: list[int],
        hidden_dims: list[int],
        dropout: float,
        embed_dim_cap: int,
    ):
        super().__init__()

        # One embedding table per categorical column. Each category value is an int index.
        self.embeddings = nn.ModuleList()
        emb_dims: list[int] = []
        for size in cat_sizes:
            # Rule-of-thumb: embedding dim grows slowly with number of categories.
            dim = min(embed_dim_cap, max(4, int(size**0.5) + 1))
            emb_dims.append(dim)
            self.embeddings.append(nn.Embedding(num_embeddings=size, embedding_dim=dim))

        # Build a simple MLP: [input -> hidden... -> 1].
        in_dim = n_num + sum(emb_dims)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # For each categorical column i, look up its embedding and concatenate.
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(x_cat[:, i]))
        x = torch.cat([x_num] + embs, dim=1)
        return self.mlp(x)


def _parse_int_list(csv_list: str) -> list[int]:
    """Parse comma-separated ints like '256,128' -> [256, 128]."""
    items = [s.strip() for s in csv_list.split(",") if s.strip()]
    if not items:
        raise ValueError("hidden-dims must contain at least one integer, e.g. 256,128")
    out: list[int] = []
    for s in items:
        out.append(int(s))
    return out


def _make_splits(n: int, train_frac: float, val_frac: float, test_frac: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Return (train_idx, val_idx, test_idx) lists using a fixed random seed."""
    if n <= 0:
        raise ValueError("Dataset is empty.")
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1):
        raise ValueError("Fractions must be between 0 and 1.")
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # Put the leftover into test to ensure all rows are used.
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Split produced an empty partition; adjust fractions.")

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _eval_mse(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    """Compute mean MSE over a DataLoader (predicting log1p(price))."""
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            pred = model(x_num.to(device), x_cat.to(device))
            total += float(loss_fn(pred, y.to(device)).item())
            count += y.shape[0]
    return total / max(1, count)


def train_and_eval(
    *,
    payload: dict,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dims: list[int],
    dropout: float,
    embed_dim_cap: int,
    seed: int,
    device: torch.device | None = None,
    compute_test: bool = True,
) -> dict:
    """
    Train the ANN once and return a dict of metrics.

    This function is intentionally small + explicit so it can be reused by:
      - `train_ann.py` (single run)
      - `sweep_ann.py` (many runs)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TensorDataset(payload)

    # For fair comparisons in a sweep, we seed:
    #   - model initialization (torch.manual_seed)
    #   - DataLoader shuffling order (generator)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = None
    if compute_test:
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)

    cat_sizes = [len(payload["cat_maps"][col]) for col in payload["cat_cols"]]

    model = PriceANN(
        n_num=payload["X_num"].shape[1],
        cat_sizes=cat_sizes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        embed_dim_cap=embed_dim_cap,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Keep the best checkpoint by validation loss (standard practice).
    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x_num, x_cat, y in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x_num.to(device), x_cat.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            opt.step()

        val_mse = _eval_mse(model, val_loader, device)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mse = _eval_mse(model, train_loader, device)
    val_mse = _eval_mse(model, val_loader, device)
    test_mse = None
    if compute_test and test_loader is not None:
        test_mse = _eval_mse(model, test_loader, device)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "device": device.type,
        "n_params": int(n_params),
        "best_val_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "test_mse": None if test_mse is None else float(test_mse),
        "train_rmse": float(train_mse**0.5),
        "val_rmse": float(val_mse**0.5),
        "test_rmse": None if test_mse is None else float(test_mse**0.5),
    }


def train_fixed_epochs_eval_test(
    *,
    payload: dict,
    train_idx: list[int],
    test_idx: list[int],
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dims: list[int],
    dropout: float,
    embed_dim_cap: int,
    seed: int,
    device: torch.device | None = None,
) -> dict:
    """
    Train for a fixed number of epochs and evaluate on test.

    This is used for a "clean" final evaluation:
      1) pick hyperparams using validation
      2) train once on (train + val) for the chosen number of epochs
      3) report test metrics exactly once
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TensorDataset(payload)

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)

    cat_sizes = [len(payload["cat_maps"][col]) for col in payload["cat_cols"]]

    model = PriceANN(
        n_num=payload["X_num"].shape[1],
        cat_sizes=cat_sizes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        embed_dim_cap=embed_dim_cap,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _epoch in range(epochs):
        model.train()
        for x_num, x_cat, y in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x_num.to(device), x_cat.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            opt.step()

    train_mse = _eval_mse(model, train_loader, device)
    test_mse = _eval_mse(model, test_loader, device)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "device": device.type,
        "n_params": int(n_params),
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "train_rmse": float(train_mse**0.5),
        "test_rmse": float(test_mse**0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors.pt")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dims", default="256,128", help="Comma-separated hidden sizes, e.g. 256,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed-dim-cap", type=int, default=64)

    # Required by you: 70/15/15 split.
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)

    # Seed affects both the split and the weight initialization.
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = torch.load(args.data, map_location="cpu")
    n = payload["X_num"].shape[0]
    train_idx, val_idx, test_idx = _make_splits(n, args.train_frac, args.val_frac, args.test_frac, args.seed)
    hidden_dims = _parse_int_list(args.hidden_dims)
    metrics = train_and_eval(
        payload=payload,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        embed_dim_cap=args.embed_dim_cap,
        seed=args.seed,
    )

    print("\nFinal summary (predicting log1p(TransactionPrice)):")
    print(f"  split: train/val/test = {args.train_frac:.2f}/{args.val_frac:.2f}/{args.test_frac:.2f}")
    print(f"  samples: n={n}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  device: {metrics['device']}")
    print(f"  hyperparams: hidden_dims={hidden_dims} dropout={args.dropout} embed_dim_cap={args.embed_dim_cap} lr={args.lr} weight_decay={args.weight_decay}")
    print(f"  n_params: {metrics['n_params']}")
    print(f"  best_val_epoch: {metrics['best_val_epoch']}  best_val_mse: {metrics['best_val_mse']:.6f}")
    print(f"  train_mse: {metrics['train_mse']:.6f}  train_rmse: {metrics['train_rmse']:.6f}")
    print(f"  val_mse:   {metrics['val_mse']:.6f}  val_rmse:   {metrics['val_rmse']:.6f}")
    print(f"  test_mse:  {metrics['test_mse']:.6f}  test_rmse:  {metrics['test_rmse']:.6f}")


if __name__ == "__main__":
    main()
