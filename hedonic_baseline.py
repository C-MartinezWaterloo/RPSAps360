#!/usr/bin/env python3
"""
Hedonic regression baseline (linear model).

This is a classic baseline for housing price modeling:
  log1p(price) = intercept
              + (numeric features * linear weights)
              + (categorical fixed-effects / one-hot coefficients)

Implementation detail:
  - We already have:
      X_num : standardized numeric features
      X_cat : integer category indices
      y_log1p : log1p(TransactionPrice)
    in `ann_tensors.pt` (created by prepare_ann_tensors.py).
  - A linear model with categorical fixed-effects is equivalent to:
      numeric linear layer + 1D embedding tables (one embedding per category).

We train on the same 70/15/15 split used elsewhere and report test error.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from train_ann import _make_splits


class HedonicDataset(Dataset):
    """Dataset wrapper around the tensors in ann_tensors.pt."""

    def __init__(self, payload: dict):
        self.x_num = payload["X_num"]
        self.x_cat = payload["X_cat"]
        self.y_log = payload["y_log1p"]
        self.y_raw = payload["y"]

    def __len__(self) -> int:
        return self.x_num.shape[0]

    def __getitem__(self, idx: int):
        return self.x_num[idx], self.x_cat[idx], self.y_log[idx], self.y_raw[idx]


class HedonicLinear(nn.Module):
    """
    Linear hedonic model:
      y_hat = b + w^T x_num + sum_i beta_i[cat_i]

    where beta_i is a per-category coefficient (fixed effect).
    """

    def __init__(self, n_num: int, cat_sizes: list[int]):
        super().__init__()
        self.num = nn.Linear(n_num, 1, bias=True)
        self.cat = nn.ModuleList([nn.Embedding(size, 1) for size in cat_sizes])

        # Start categorical effects at 0 (so the model begins as just numeric linear).
        for emb in self.cat:
            nn.init.zeros_(emb.weight)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        out = self.num(x_num)
        for i, emb in enumerate(self.cat):
            out = out + emb(x_cat[:, i])
        return out


def _eval_log_mse_and_rmse(model: nn.Module, loader: Iterable, device: torch.device) -> tuple[float, float]:
    """Evaluate MSE/RMSE on log1p(price)."""
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for x_num, x_cat, y_log, _y_raw in loader:
            pred = model(x_num.to(device), x_cat.to(device))
            diff = pred - y_log.to(device)
            loss_sum += float((diff * diff).sum().item())
            count += y_log.shape[0]
    mse = loss_sum / max(1, count)
    rmse = mse**0.5
    return mse, rmse


def _eval_price_rmse(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    """
    Evaluate RMSE in dollars by converting:
      price_pred = expm1(y_hat_log)

    Note: this is *not* what we train on; it's just an interpretable metric.
    """
    model.eval()
    se_sum = 0.0
    count = 0
    with torch.no_grad():
        for x_num, x_cat, _y_log, y_raw in loader:
            pred_log = model(x_num.to(device), x_cat.to(device))
            pred_price = torch.expm1(pred_log).clamp_min(0.0)
            diff = pred_price - y_raw.to(device)
            se_sum += float((diff * diff).sum().item())
            count += y_raw.shape[0]
    mse = se_sum / max(1, count)
    return mse**0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional: append a single summary row to this CSV (useful for Excel export).",
    )
    args = parser.parse_args()

    start_time = time.time()

    payload = torch.load(args.data, map_location="cpu")
    ds = HedonicDataset(payload)

    # Same split scheme as the ANN scripts.
    n = len(ds)
    train_idx, val_idx, test_idx = _make_splits(n, 0.70, 0.15, 0.15, args.seed)

    # DataLoaders: shuffle train, keep val/test deterministic.
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=args.batch_size, shuffle=False)

    cat_sizes = [len(payload["cat_maps"][col]) for col in payload["cat_cols"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = HedonicLinear(n_num=payload["X_num"].shape[1], cat_sizes=cat_sizes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x_num, x_cat, y_log, _y_raw in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x_num.to(device), x_cat.to(device))
            loss = loss_fn(pred, y_log.to(device))
            loss.backward()
            opt.step()

        val_mse, val_rmse = _eval_log_mse_and_rmse(model, val_loader, device)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        # Light progress print (keep it readable).
        print(f"epoch {epoch}/{args.epochs}  val_rmse(log1p_price)={val_rmse:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mse, train_rmse = _eval_log_mse_and_rmse(model, train_loader, device)
    val_mse, val_rmse = _eval_log_mse_and_rmse(model, val_loader, device)
    test_mse, test_rmse = _eval_log_mse_and_rmse(model, test_loader, device)
    test_price_rmse = _eval_price_rmse(model, test_loader, device)

    n_params = sum(p.numel() for p in model.parameters())
    seconds = time.time() - start_time

    print("\nHedonic regression baseline summary:")
    print("  target: log1p(TransactionPrice)")
    print(f"  samples: n={n}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  device: {device.type}")
    print(f"  hyperparams: epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} weight_decay={args.weight_decay}")
    print(f"  n_params: {n_params}")
    print(f"  best_val_epoch: {best_epoch}  best_val_mse: {best_val:.6f}")
    print(f"  train_rmse(log): {train_rmse:.6f}  train_mse(log): {train_mse:.6f}")
    print(f"  val_rmse(log):   {val_rmse:.6f}  val_mse(log):   {val_mse:.6f}")
    print(f"  test_rmse(log):  {test_rmse:.6f}  test_mse(log):  {test_mse:.6f}")
    print(f"  test_rmse($):    {test_price_rmse:,.2f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        write_header = not out_path.exists()
        # Keep a consistent schema so export_results_excel.py can ingest it.
        row = {
            "source": "hedonic",
            "stage": "hedonic_baseline",
            "run": 1,
            "seed": args.seed,
            "split_seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "model": "hedonic_linear_fixed_effects",
            "n_params": int(n_params),
            "best_val_epoch": int(best_epoch),
            "best_val_mse": float(best_val),
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "test_mse": float(test_mse),
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "test_rmse": float(test_rmse),
            "test_price_rmse": float(test_price_rmse),
            "seconds": round(seconds, 2),
        }

        with out_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print("Wrote CSV row:", out_path)


if __name__ == "__main__":
    main()
