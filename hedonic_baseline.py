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
import random
import time
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from csv_utils import append_row
from train_ann import make_splits


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


def _eval_metrics(model: nn.Module, loader: Iterable, device: torch.device) -> dict[str, float]:
    """
    Match the ANN metric schema:
      - MSE/RMSE in log1p(price) space
      - MdAPE in price space
      - Accuracy% in price space: max(0, 100*(1 - MdAPE))
    """

    model.eval()
    mse_sum = 0.0
    n = 0
    ape_chunks: list[torch.Tensor] = []

    eps = 1e-8
    with torch.no_grad():
        for x_num, x_cat, y_log, y_raw in loader:
            pred_log = model(x_num.to(device), x_cat.to(device))
            diff = pred_log - y_log.to(device)
            mse_sum += float((diff * diff).sum().item())

            y_true = y_raw.to(device)
            y_pred = torch.expm1(pred_log).clamp_min(0.0)
            ape = (y_pred - y_true).abs() / y_true.clamp_min(eps)
            ape_chunks.append(ape.detach().cpu().view(-1))

            n += int(y_log.shape[0])

    mse = mse_sum / max(1, n)
    rmse = mse**0.5

    if ape_chunks:
        ape_all = torch.cat(ape_chunks, dim=0)
        mdape = float(torch.median(ape_all).item())
    else:
        mdape = float("nan")

    acc_pct = max(0.0, 100.0 * (1.0 - mdape)) if mdape == mdape else float("nan")

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mdape": float(mdape),
        "acc_pct": float(acc_pct),
    }


def train_and_eval(
    *,
    payload: dict,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device | None = None,
    compute_test: bool = True,
) -> dict:
    """
    Train the hedonic baseline once and return a dict of metrics.

    This mirrors the `train_ann.train_and_eval()` interface so we can run
    consistent multi-seed evaluations.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = HedonicDataset(payload)

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = None
    if compute_test:
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)

    cat_sizes = [len(payload["cat_maps"][col]) for col in payload["cat_cols"]]
    model = HedonicLinear(n_num=payload["X_num"].shape[1], cat_sizes=cat_sizes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x_num, x_cat, y_log, _y_raw in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x_num.to(device), x_cat.to(device))
            loss = loss_fn(pred, y_log.to(device))
            loss.backward()
            opt.step()

        val_mse, _val_rmse = _eval_log_mse_and_rmse(model, val_loader, device)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Metrics at the final epoch (overfitting signal).
    train_eval_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=False)
    last_train = _eval_metrics(model, train_eval_loader, device)
    last_val = _eval_metrics(model, val_loader, device)
    last_test = None
    if compute_test and test_loader is not None:
        last_test = _eval_metrics(model, test_loader, device)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = _eval_metrics(model, train_eval_loader, device)
    val_metrics = _eval_metrics(model, val_loader, device)
    test_metrics = None
    test_price_rmse = None
    if compute_test and test_loader is not None:
        test_metrics = _eval_metrics(model, test_loader, device)
        test_price_rmse = _eval_price_rmse(model, test_loader, device)

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "device": device.type,
        "n_params": int(n_params),
        "best_val_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "train_mse": float(train_metrics["mse"]),
        "val_mse": float(val_metrics["mse"]),
        "test_mse": None if test_metrics is None else float(test_metrics["mse"]),
        "train_rmse": float(train_metrics["rmse"]),
        "val_rmse": float(val_metrics["rmse"]),
        "test_rmse": None if test_metrics is None else float(test_metrics["rmse"]),
        "train_mdape": float(train_metrics["mdape"]),
        "val_mdape": float(val_metrics["mdape"]),
        "test_mdape": None if test_metrics is None else float(test_metrics["mdape"]),
        "train_acc_pct": float(train_metrics["acc_pct"]),
        "val_acc_pct": float(val_metrics["acc_pct"]),
        "test_acc_pct": None if test_metrics is None else float(test_metrics["acc_pct"]),
        "train_mse_last": float(last_train["mse"]),
        "val_mse_last": float(last_val["mse"]),
        "test_mse_last": None if last_test is None else float(last_test["mse"]),
        "train_rmse_last": float(last_train["rmse"]),
        "val_rmse_last": float(last_val["rmse"]),
        "test_rmse_last": None if last_test is None else float(last_test["rmse"]),
        "train_mdape_last": float(last_train["mdape"]),
        "val_mdape_last": float(last_val["mdape"]),
        "test_mdape_last": None if last_test is None else float(last_test["mdape"]),
        "train_acc_pct_last": float(last_train["acc_pct"]),
        "val_acc_pct_last": float(last_val["acc_pct"]),
        "test_acc_pct_last": None if last_test is None else float(last_test["acc_pct"]),
        "test_price_rmse": None if test_price_rmse is None else float(test_price_rmse),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=None,
        help="If set, train on a fixed random subset of the training split for speed; val/test stay full.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random", "time"],
        default="random",
        help="How to split data into train/val/test (default: random). 'time' sorts by TransactionYear/Quarter.",
    )
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
    train_idx_full, val_idx, test_idx = make_splits(
        payload=payload,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        seed=args.seed,
        strategy=args.split_strategy,
    )
    n_train_full = len(train_idx_full)
    train_idx = train_idx_full
    if args.train_max_samples is not None:
        if args.train_max_samples <= 0:
            raise ValueError("--train-max-samples must be positive.")
        if args.train_max_samples < len(train_idx_full):
            rng_sub = random.Random(args.seed)
            train_idx = rng_sub.sample(train_idx_full, args.train_max_samples)
    n_train_used = len(train_idx)

    # DataLoaders: shuffle train, keep val/test deterministic.
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = train_and_eval(
        payload=payload,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        compute_test=True,
    )
    seconds = time.time() - start_time

    print("\nHedonic regression baseline summary:")
    print("  target: log1p(TransactionPrice)")
    extra = f" (train_max_samples={args.train_max_samples})" if args.train_max_samples is not None else ""
    print(f"  samples: n={n}  train={len(train_idx)} / {n_train_full}{extra}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  device: {metrics.get('device')}")
    print(f"  split_strategy: {args.split_strategy}")
    print(f"  hyperparams: epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} weight_decay={args.weight_decay}")
    print(f"  n_params: {metrics['n_params']}")
    print(f"  best_val_epoch: {metrics['best_val_epoch']}  best_val_mse: {metrics['best_val_mse']:.6f}")
    print(f"  train_rmse(log): {metrics['train_rmse']:.6f}  train_mse(log): {metrics['train_mse']:.6f}")
    print(f"  val_rmse(log):   {metrics['val_rmse']:.6f}  val_mse(log):   {metrics['val_mse']:.6f}")
    print(f"  test_rmse(log):  {metrics['test_rmse']:.6f}  test_mse(log):  {metrics['test_mse']:.6f}")
    if metrics.get("test_price_rmse") is not None:
        print(f"  test_rmse($):    {float(metrics['test_price_rmse']):,.2f}")
    print(
        "  acc_pct (best checkpoint):"
        f" train={metrics['train_acc_pct']:.2f}%"
        f" val={metrics['val_acc_pct']:.2f}%"
        f" test={metrics['test_acc_pct']:.2f}%"
    )
    print(
        "  acc_pct (last epoch):"
        f" train={metrics['train_acc_pct_last']:.2f}%"
        f" val={metrics['val_acc_pct_last']:.2f}%"
        f" test={metrics['test_acc_pct_last']:.2f}%"
    )

    if args.out_csv:
        out_path = Path(args.out_csv)
        # Keep a consistent schema so export_results_excel.py can ingest it.
        row = {
            "source": "hedonic",
            "stage": "hedonic_baseline",
            "run": 1,
            # Record the tensor file + feature set so results are comparable in `results_all.csv`.
            "data": args.data,
            "feature_set": payload.get("feature_set", ""),
            "seed": args.seed,
            "split_seed": args.seed,
            "split_strategy": args.split_strategy,
            "train_max_samples": "" if args.train_max_samples is None else int(args.train_max_samples),
            "n_train_full": int(n_train_full),
            "n_train_used": int(n_train_used),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "model": "hedonic_linear_fixed_effects",
            "n_params": metrics["n_params"],
            "best_val_epoch": metrics["best_val_epoch"],
            "best_val_mse": metrics["best_val_mse"],
            "train_mse": metrics["train_mse"],
            "val_mse": metrics["val_mse"],
            "test_mse": metrics["test_mse"],
            "train_rmse": metrics["train_rmse"],
            "val_rmse": metrics["val_rmse"],
            "test_rmse": metrics["test_rmse"],
            "train_mdape": metrics["train_mdape"],
            "val_mdape": metrics["val_mdape"],
            "test_mdape": metrics["test_mdape"],
            "train_acc_pct": metrics["train_acc_pct"],
            "val_acc_pct": metrics["val_acc_pct"],
            "test_acc_pct": metrics["test_acc_pct"],
            "train_mse_last": metrics["train_mse_last"],
            "val_mse_last": metrics["val_mse_last"],
            "test_mse_last": metrics["test_mse_last"],
            "train_rmse_last": metrics["train_rmse_last"],
            "val_rmse_last": metrics["val_rmse_last"],
            "test_rmse_last": metrics["test_rmse_last"],
            "train_mdape_last": metrics["train_mdape_last"],
            "val_mdape_last": metrics["val_mdape_last"],
            "test_mdape_last": metrics["test_mdape_last"],
            "train_acc_pct_last": metrics["train_acc_pct_last"],
            "val_acc_pct_last": metrics["val_acc_pct_last"],
            "test_acc_pct_last": metrics["test_acc_pct_last"],
            "test_price_rmse": metrics["test_price_rmse"],
            "seconds": round(seconds, 2),
        }
        append_row(out_path, row)
        print("Wrote CSV row:", out_path)


if __name__ == "__main__":
    main()
