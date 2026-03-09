#!/usr/bin/env python3
"""
Train a TabNet regressor on prepared tensors (numeric + categorical embeddings).

Notes:
  - This is a dependency-free (PyTorch-only) TabNet-style model.
  - We embed categoricals, concatenate with numeric features, then apply
    sequential feature selection with sparse masks (sparsemax).

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
import math
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


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation (Martins & Astudillo, 2016).

    Projects logits onto the probability simplex, producing sparse probabilities.
    """

    if dim < 0:
        dim += logits.dim()

    z = logits
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = z_sorted.cumsum(dim)

    r = torch.arange(1, z_sorted.size(dim) + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.dim()
    view[dim] = -1
    r = r.view(view)

    # Determine k(z): max k such that 1 + k z_k > sum_{i<=k} z_i
    support = 1 + r * z_sorted > z_cumsum
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)

    # Compute tau = (sum_{i<=k} z_i - 1) / k
    z_cumsum_k = z_cumsum.gather(dim, k - 1)
    tau = (z_cumsum_k - 1) / k.to(z.dtype)

    return torch.clamp(z - tau, min=0.0)


class CatEmbedder(nn.Module):
    def __init__(self, *, cat_sizes: list[int], embed_dim_cap: int, dropout: float):
        super().__init__()
        self.embs = nn.ModuleList()
        self.dims: list[int] = []
        for size in cat_sizes:
            dim = min(embed_dim_cap, max(4, int(size**0.5) + 1))
            self.dims.append(dim)
            self.embs.append(nn.Embedding(num_embeddings=size, embedding_dim=dim))
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    @property
    def out_dim(self) -> int:
        return int(sum(self.dims))

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if not self.embs:
            return torch.empty((x_cat.shape[0], 0), device=x_cat.device, dtype=torch.float32)
        out = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        return self.drop(out)


class GLULayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim * 2)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        out, gate = x.chunk(2, dim=1)
        out = out * torch.sigmoid(gate)
        return self.drop(out)


class GLUBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = in_dim
        for _ in range(n_layers):
            layers.append(GLULayer(d_in, out_dim, dropout))
            d_in = out_dim
        self.layers = nn.ModuleList(layers)
        self.scale = math.sqrt(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            y = layer(out)
            out = (out + y) * self.scale if out.shape == y.shape else y
        return out


class FeatureTransformer(nn.Module):
    def __init__(self, *, in_dim: int, out_dim: int, n_shared: int, n_independent: int, dropout: float):
        super().__init__()

        # Shared blocks applied at every step.
        shared_layers: list[nn.Module] = []
        d_in = in_dim
        for _ in range(n_shared):
            shared_layers.append(GLUBlock(d_in, out_dim, n_layers=1, dropout=dropout))
            d_in = out_dim
        self.shared = nn.ModuleList(shared_layers)

        # Step-specific independent blocks.
        indep_layers: list[nn.Module] = []
        d_in = out_dim if n_shared > 0 else in_dim
        for _ in range(n_independent):
            indep_layers.append(GLUBlock(d_in, out_dim, n_layers=1, dropout=dropout))
            d_in = out_dim
        self.independent = nn.ModuleList(indep_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.shared:
            out = layer(out)
        for layer in self.independent:
            out = layer(out)
        return out


class AttentiveTransformer(nn.Module):
    def __init__(self, *, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, a: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        x = self.fc(a)
        x = self.bn(x)
        x = x * prior
        return sparsemax(x, dim=1)


class TabNetRegressor(nn.Module):
    def __init__(
        self,
        *,
        n_num: int,
        cat_sizes: list[int],
        embed_dim_cap: int,
        cat_dropout: float,
        n_d: int,
        n_a: int,
        n_steps: int,
        gamma: float,
        n_shared: int,
        n_independent: int,
        dropout: float,
    ):
        super().__init__()

        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if gamma <= 1.0:
            raise ValueError("gamma must be > 1.0.")

        self.emb = CatEmbedder(cat_sizes=cat_sizes, embed_dim_cap=embed_dim_cap, dropout=cat_dropout)
        self.n_features = int(n_num + self.emb.out_dim)
        self.n_groups = int(n_num + len(cat_sizes))
        self.register_buffer(
            "group_sizes",
            torch.tensor(([1] * int(n_num)) + list(self.emb.dims), dtype=torch.long),
        )

        self.input_bn = nn.BatchNorm1d(self.n_features)

        self.n_d = int(n_d)
        self.n_a = int(n_a)
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)

        out_dim = self.n_d + self.n_a
        self.ft_shared = FeatureTransformer(
            in_dim=self.n_features, out_dim=out_dim, n_shared=n_shared, n_independent=0, dropout=dropout
        )
        self.ft_initial = FeatureTransformer(
            in_dim=out_dim, out_dim=out_dim, n_shared=0, n_independent=n_independent, dropout=dropout
        )
        self.ft_steps = nn.ModuleList(
            [
                FeatureTransformer(in_dim=out_dim, out_dim=out_dim, n_shared=0, n_independent=n_independent, dropout=dropout)
                for _ in range(self.n_steps)
            ]
        )
        self.att_steps = nn.ModuleList([AttentiveTransformer(in_dim=self.n_a, out_dim=self.n_groups) for _ in range(self.n_steps)])

        self.fc_out = nn.Linear(self.n_d, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, *, return_sparse_loss: bool = False):
        x = torch.cat([x_num, self.emb(x_cat)], dim=1)
        x = self.input_bn(x)

        # Group-level prior/masks:
        #   - numeric: 1 group per numeric feature (size=1)
        #   - categorical: 1 group per categorical column (size=embed_dim)
        # This avoids masking individual embedding dimensions, which is both
        # harder to interpret and empirically unstable.
        prior = torch.ones((x.shape[0], self.n_groups), device=x.device, dtype=x.dtype)
        sparse_loss = 0.0

        # Initial splitter creates the first attention embedding `a` (TabNet-style).
        out0 = self.ft_shared(x)
        out0 = self.ft_initial(out0)
        _, a = out0.split([self.n_d, self.n_a], dim=1)
        decision_sum = torch.zeros((x.shape[0], self.n_d), device=x.device, dtype=x.dtype)

        eps = 1e-8
        for step in range(self.n_steps):
            mask_group = self.att_steps[step](a, prior)
            prior = prior * (self.gamma - mask_group)

            if return_sparse_loss:
                entropy = (-mask_group * torch.log(mask_group.clamp_min(eps))).sum(dim=1).mean()
                sparse_loss = sparse_loss + entropy

            mask = mask_group.repeat_interleave(self.group_sizes, dim=1)
            x_masked = mask * x

            out = self.ft_shared(x_masked)
            out = self.ft_steps[step](out)
            d, a = out.split([self.n_d, self.n_a], dim=1)
            decision_sum = decision_sum + torch.relu(d)

        pred = self.fc_out(decision_sum)
        if return_sparse_loss:
            return pred, sparse_loss / max(1, self.n_steps)
        return pred


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
    embed_dim_cap: int,
    cat_dropout: float,
    n_d: int,
    n_a: int,
    n_steps: int,
    gamma: float,
    n_shared: int,
    n_independent: int,
    dropout: float,
    lambda_sparse: float,
    seed: int,
    val_max_samples: int | None = None,
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
    val_idx_select = val_idx
    if val_max_samples is not None and val_max_samples > 0 and val_max_samples < len(val_idx):
        val_idx_select = val_idx[:val_max_samples]
    val_loader_select = DataLoader(Subset(ds, val_idx_select), batch_size=batch_size, shuffle=False)
    test_loader = None
    if compute_test:
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)

    cat_sizes = [len(payload["cat_maps"][col]) for col in payload["cat_cols"]]

    model = TabNetRegressor(
        n_num=payload["X_num"].shape[1],
        cat_sizes=cat_sizes,
        embed_dim_cap=embed_dim_cap,
        cat_dropout=cat_dropout,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        n_shared=n_shared,
        n_independent=n_independent,
        dropout=dropout,
    ).to(device)

    # Stabilize training for large-batch sweeps by initializing the output bias
    # to the (log1p) target mean on the training split. Without this, the model
    # starts near 0 while targets are ~13, and AdamW updates the bias very
    # slowly when steps-per-epoch is small.
    with torch.no_grad():
        y_mean = float(payload["y_log1p"][train_idx].mean().item())
        model.fc_out.bias.data.fill_(y_mean)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_select = float("inf")
    best_epoch = 0
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x_num, x_cat, y in train_loader:
            opt.zero_grad(set_to_none=True)
            pred, sparse_loss = model(x_num.to(device), x_cat.to(device), return_sparse_loss=True)
            mse = loss_fn(pred, y.to(device))
            loss = mse + float(lambda_sparse) * sparse_loss
            loss.backward()
            opt.step()

        val_mse_select = _eval_mse(model, val_loader_select, device)
        if val_mse_select < best_val_select:
            best_val_select = val_mse_select
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
        # Always report best_val_mse on the full validation set (even if epoch
        # selection used a smaller subset for speed).
        "best_val_mse": float(val_metrics["mse"]),
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
    parser.add_argument("--embed-dim-cap", type=int, default=64)
    parser.add_argument("--cat-dropout", type=float, default=0.0)

    parser.add_argument("--n-d", type=int, default=32)
    parser.add_argument("--n-a", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--n-shared", type=int, default=1)
    parser.add_argument("--n-independent", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lambda-sparse", type=float, default=1e-5)

    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-3)
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

    parser.add_argument("--run-name", default="tabnet_single")
    parser.add_argument("--out-csv", default=None)
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

    metrics = train_and_eval(
        payload=payload,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim_cap=args.embed_dim_cap,
        cat_dropout=args.cat_dropout,
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=args.gamma,
        n_shared=args.n_shared,
        n_independent=args.n_independent,
        dropout=args.dropout,
        lambda_sparse=args.lambda_sparse,
        seed=args.seed,
        compute_test=True,
    )

    seconds = time.time() - start_time

    print("\nTabNet summary:")
    print("  target: log1p(TransactionPrice)")
    print(f"  samples: n={len(ds)}  train={n_train_used}/{n_train_full}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  device: {metrics.get('device')}")
    print(f"  split_strategy: {args.split_strategy}")
    print(
        "  hyperparams:"
        f" n_d={args.n_d} n_a={args.n_a} n_steps={args.n_steps} gamma={args.gamma}"
        f" shared={args.n_shared} indep={args.n_independent}"
        f" embed_dim_cap={args.embed_dim_cap} cat_dropout={args.cat_dropout}"
        f" dropout={args.dropout} lambda_sparse={args.lambda_sparse}"
        f" epochs={args.epochs} batch_size={args.batch_size}"
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
            "source": "train_tabnet",
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
            "embed_dim_cap": args.embed_dim_cap,
            "cat_dropout": args.cat_dropout,
            "n_d": args.n_d,
            "n_a": args.n_a,
            "n_steps": args.n_steps,
            "gamma": args.gamma,
            "n_shared": args.n_shared,
            "n_independent": args.n_independent,
            "lambda_sparse": args.lambda_sparse,
            "model": "tabnet",
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
