#!/usr/bin/env python3
"""
Train a DeepFM model (linear + factorization machine interactions + deep MLP).

Why DeepFM:
  - Hedonic is linear.
  - FM captures pairwise interactions but stays shallow.
  - ANN captures nonlinearities but may need more data/training to generalize well.
  - DeepFM combines:
      1) linear terms
      2) low-rank pairwise interactions (FM)
      3) a deep MLP on embeddings + numeric features

Target:
  - y_log1p = log1p(TransactionPrice)

Metrics:
  - MSE/RMSE in log1p(price) space (RMSLE-like)
  - MdAPE in price space (median absolute percentage error)
  - Accuracy% in price space: max(0, 100*(1 - MdAPE))
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


class TensorDataset(Dataset):
    """Wrapper around tensors saved by prepare_ann_tensors.py."""

    def __init__(self, payload: dict):
        self.x_num = payload["X_num"]
        self.x_cat = payload["X_cat"]
        self.y = payload["y_log1p"]

    def __len__(self) -> int:
        return self.x_num.shape[0]

    def __getitem__(self, idx: int):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


def _parse_int_list(csv_list: str) -> list[int]:
    items = [s.strip() for s in csv_list.split(",") if s.strip()]
    if not items:
        raise ValueError("--hidden-dims must contain at least one integer, e.g. 512,256,128")
    return [int(s) for s in items]


class DeepFM(nn.Module):
    def __init__(
        self,
        *,
        n_num: int,
        cat_sizes: list[int],
        factor_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()

        if factor_dim <= 0:
            raise ValueError("factor_dim must be positive.")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty.")

        self.bias = nn.Parameter(torch.zeros(1))

        # Linear terms.
        self.linear_num = nn.Linear(n_num, 1, bias=False)
        self.linear_cat = nn.ModuleList([nn.Embedding(size, 1) for size in cat_sizes])
        for emb in self.linear_cat:
            nn.init.zeros_(emb.weight)

        # Factor terms (interaction vectors) for FM part.
        self.v_num = nn.Parameter(torch.randn(n_num, factor_dim) * 0.01)
        self.v_cat = nn.ModuleList([nn.Embedding(size, factor_dim) for size in cat_sizes])
        for emb in self.v_cat:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

        self.factor_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Deep part: numeric + flattened categorical factor embeddings.
        deep_in = n_num + len(cat_sizes) * factor_dim
        layers: list[nn.Module] = []
        prev = deep_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # Linear part.
        out = self.bias + self.linear_num(x_num)
        if self.linear_cat:
            for i, emb in enumerate(self.linear_cat):
                out = out + emb(x_cat[:, i])

        # FM interaction part.
        v_num = x_num.unsqueeze(2) * self.v_num.unsqueeze(0)  # [B, n_num, k]
        if self.v_cat:
            v_cat = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.v_cat)], dim=1)  # [B, n_cat, k]
            v = torch.cat([v_num, v_cat], dim=1)  # [B, n_features, k]
        else:
            v_cat = None
            v = v_num

        v = self.factor_drop(v)
        sum_v = v.sum(dim=1)  # [B, k]
        interactions = 0.5 * ((sum_v * sum_v - (v * v).sum(dim=1)).sum(dim=1, keepdim=True))  # [B, 1]

        # Deep part.
        if v_cat is None:
            deep_in = x_num
        else:
            deep_in = torch.cat([x_num, v_cat.reshape(v_cat.shape[0], -1)], dim=1)
        deep_out = self.deep(deep_in)

        return out + interactions + deep_out


def _eval_mse(model: nn.Module, loader: Iterable, device: torch.device) -> float:
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


def _eval_metrics(model: nn.Module, loader: Iterable, device: torch.device) -> dict[str, float]:
    model.eval()
    mse_sum_fn = nn.MSELoss(reduction="sum")

    mse_sum = 0.0
    n = 0
    ape_chunks: list[torch.Tensor] = []

    eps = 1e-8
    with torch.no_grad():
        for x_num, x_cat, y_log1p in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y_log1p = y_log1p.to(device)

            pred_log1p = model(x_num, x_cat)
            mse_sum += float(mse_sum_fn(pred_log1p, y_log1p).item())

            y_true = torch.expm1(y_log1p)
            y_pred = torch.expm1(pred_log1p)
            ape = (y_pred - y_true).abs() / y_true.clamp_min(eps)
            ape_chunks.append(ape.detach().cpu().view(-1))
            n += int(y_true.shape[0])

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
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    factor_dim: int,
    hidden_dims: list[int],
    dropout: float,
    seed: int,
    device: torch.device | None = None,
    compute_test: bool = True,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TensorDataset(payload)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = None
    if compute_test:
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)

    cat_sizes = [len(payload["cat_maps"][col]) for col in payload["cat_cols"]]

    model = DeepFM(
        n_num=payload["X_num"].shape[1],
        cat_sizes=cat_sizes,
        factor_dim=factor_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

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
    if compute_test and test_loader is not None:
        test_metrics = _eval_metrics(model, test_loader, device)

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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors_full.pt")
    parser.add_argument("--factor-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", default="512,256,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
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

    parser.add_argument("--run-name", default="deepfm_single")
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional: append a single summary row to this CSV (gitignored; merged into results_all.csv).",
    )
    args = parser.parse_args()

    start_time = time.time()

    payload = torch.load(args.data, map_location="cpu")
    ds = TensorDataset(payload)

    train_idx_full, val_idx, test_idx = make_splits(
        payload=payload,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
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
        factor_dim=args.factor_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        seed=args.seed,
        compute_test=True,
    )

    seconds = time.time() - start_time

    print("\nDeepFM summary:")
    print("  target: log1p(TransactionPrice)")
    print(f"  samples: n={len(ds)}  train={n_train_used}/{n_train_full}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  device: {metrics.get('device')}")
    print(f"  split_strategy: {args.split_strategy}")
    print(
        "  hyperparams:"
        f" factor_dim={args.factor_dim} hidden_dims={','.join(str(x) for x in hidden_dims)}"
        f" dropout={args.dropout} epochs={args.epochs} batch_size={args.batch_size}"
        f" lr={args.lr} weight_decay={args.weight_decay}"
    )
    print(f"  n_params: {metrics['n_params']}")
    print(f"  best_val_epoch: {metrics['best_val_epoch']}  best_val_mse: {metrics['best_val_mse']:.6f}")
    print(f"  train_acc_pct: {metrics['train_acc_pct']:.2f}  val_acc_pct: {metrics['val_acc_pct']:.2f}  test_acc_pct: {metrics['test_acc_pct']:.2f}")
    print(
        "  last_epoch_acc_pct:"
        f" train={metrics['train_acc_pct_last']:.2f}"
        f" val={metrics['val_acc_pct_last']:.2f}"
        f" test={metrics['test_acc_pct_last']:.2f}"
    )
    print(f"  seconds: {seconds:.2f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        row = {
            "source": "train_deepfm",
            "stage": "single_run",
            "run_name": args.run_name,
            "data": args.data,
            "feature_set": payload.get("feature_set", ""),
            "seed": args.seed,
            "split_seed": args.seed,
            "split_strategy": args.split_strategy,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "train_max_samples": "" if args.train_max_samples is None else int(args.train_max_samples),
            "n_train_full": int(n_train_full),
            "n_train_used": int(n_train_used),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "factor_dim": args.factor_dim,
            "hidden_dims": ",".join(str(x) for x in hidden_dims),
            "model": "deepfm",
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
            "seconds": round(seconds, 2),
        }
        append_row(out_path, row)
        print("Wrote CSV row:", out_path)


if __name__ == "__main__":
    main()

