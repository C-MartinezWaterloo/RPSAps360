#!/usr/bin/env python3
"""
Train a TabTransformer-style model on prepared tensors (categorical attention + MLP).

Why TabTransformer:
  - A strong modern tabular baseline that applies self-attention over categorical
    feature embeddings to model cross-feature interactions.
  - Complementary to FT-Transformer: attention is restricted to categoricals,
    while numerics are handled by an MLP projection.

Data + target:
  - Inputs come from `prepare_ann_tensors.py` outputs (e.g. `ann_tensors_full.pt`)
  - Target is `y_log1p = log1p(TransactionPrice)`

Metrics (consistent with the rest of the repo):
  - MSE/RMSE in log1p(price) space
  - MdAPE + Accuracy% in price space, where accuracy% = max(0, 100*(1 - MdAPE))
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


def _parse_int_list(csv_list: str) -> list[int]:
    items = [s.strip() for s in csv_list.split(",") if s.strip()]
    if not items:
        raise ValueError("mlp-hidden-dims must contain at least one integer, e.g. 256,128")
    return [int(s) for s in items]


class CategoricalTokenizer(nn.Module):
    """One embedding table per categorical column, all with embedding_dim=d_model."""

    def __init__(self, *, cat_sizes: list[int], d_model: int):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(size, d_model) for size in cat_sizes])
        for emb in self.embs:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if not self.embs:
            return torch.empty((x_cat.shape[0], 0, 0), device=x_cat.device, dtype=torch.float32)
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return torch.stack(tokens, dim=1)  # [B, n_cat, d_model]


class TabTransformerRegressor(nn.Module):
    """
    TabTransformer-style regressor:
      - embed + contextualize categorical tokens with a Transformer encoder
      - project all numeric features into a single vector
      - concatenate [CLS(categoricals), proj(numerics)] and predict with an MLP head
    """

    def __init__(
        self,
        *,
        n_num: int,
        cat_sizes: list[int],
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
        mlp_hidden_dims: list[int],
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if ff_mult <= 0:
            raise ValueError("ff_mult must be positive.")

        self.cat_tok = CategoricalTokenizer(cat_sizes=cat_sizes, d_model=d_model)
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        # Per-position embeddings for [CLS] + each categorical column.
        self.pos_emb = nn.Parameter(torch.empty(1 + len(cat_sizes), d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        ff_dim = int(ff_mult * d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.num_proj = nn.Sequential(
            nn.Linear(n_num, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        head_in = int(2 * d_model)
        head_layers: list[nn.Module] = [nn.LayerNorm(head_in)]
        prev = head_in
        for h in mlp_hidden_dims:
            head_layers.append(nn.Linear(prev, h))
            head_layers.append(nn.ReLU())
            if dropout and dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            prev = h
        head_layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        cat_tokens = self.cat_tok(x_cat)  # [B, n_cat, d]
        cls = self.cls.expand(cat_tokens.shape[0], -1, -1)  # [B, 1, d]
        x = torch.cat([cls, cat_tokens], dim=1)  # [B, 1+n_cat, d]
        x = x + self.pos_emb.unsqueeze(0)

        x = self.encoder(x)
        cls_out = x[:, 0, :]  # [B, d]
        num_out = self.num_proj(x_num)  # [B, d]

        h = torch.cat([cls_out, num_out], dim=1)  # [B, 2d]
        return self.head(h)


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
    rmse = math.sqrt(mse)

    if ape_chunks:
        ape_all = torch.cat(ape_chunks, dim=0)
        mdape = float(torch.median(ape_all).item())
    else:
        mdape = float("nan")

    acc_pct = float(max(0.0, 100.0 * (1.0 - mdape))) if mdape == mdape else float("nan")
    return {"mse": float(mse), "rmse": float(rmse), "mdape": float(mdape), "acc_pct": float(acc_pct)}


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
    d_model: int,
    n_heads: int,
    n_layers: int,
    ff_mult: int,
    dropout: float,
    mlp_hidden_dims: list[int],
    seed: int,
    clip_grad: float | None = None,
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
    val_idx_ckpt = val_idx
    if val_max_samples is not None and val_max_samples > 0 and val_max_samples < len(val_idx):
        rng_val = random.Random(seed + 1337)
        val_idx_ckpt = rng_val.sample(val_idx, val_max_samples)
    val_loader_ckpt = DataLoader(Subset(ds, val_idx_ckpt), batch_size=batch_size, shuffle=False)
    test_loader = None
    if compute_test:
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)

    cat_sizes = [len(payload["cat_maps"][col]) for col in payload["cat_cols"]]
    model = TabTransformerRegressor(
        n_num=int(payload["X_num"].shape[1]),
        cat_sizes=cat_sizes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=ff_mult,
        dropout=dropout,
        mlp_hidden_dims=mlp_hidden_dims,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 1
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x_num, x_cat, y in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x_num.to(device), x_cat.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            if clip_grad is not None and clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad))
            opt.step()

        val_mse = _eval_mse(model, val_loader_ckpt, device)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Metrics at the last epoch (overfitting check).
    train_eval_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=False)
    last_train_metrics = _eval_metrics(model, train_eval_loader, device)
    last_val_metrics = _eval_metrics(model, val_loader, device)
    last_test_metrics = None
    if compute_test and test_loader is not None:
        last_test_metrics = _eval_metrics(model, test_loader, device)

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
        "train_mse_last": float(last_train_metrics["mse"]),
        "val_mse_last": float(last_val_metrics["mse"]),
        "test_mse_last": None if last_test_metrics is None else float(last_test_metrics["mse"]),
        "train_rmse_last": float(last_train_metrics["rmse"]),
        "val_rmse_last": float(last_val_metrics["rmse"]),
        "test_rmse_last": None if last_test_metrics is None else float(last_test_metrics["rmse"]),
        "train_mdape_last": float(last_train_metrics["mdape"]),
        "val_mdape_last": float(last_val_metrics["mdape"]),
        "test_mdape_last": None if last_test_metrics is None else float(last_test_metrics["mdape"]),
        "train_acc_pct_last": float(last_train_metrics["acc_pct"]),
        "val_acc_pct_last": float(last_val_metrics["acc_pct"]),
        "test_acc_pct_last": None if last_test_metrics is None else float(last_test_metrics["acc_pct"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors_full.pt")

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-dims", default="256,128")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument(
        "--val-max-samples",
        type=int,
        default=10_000,
        help="If set, select best epoch using only a fixed random subset of validation rows (faster on CPU).",
    )

    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-max-samples", type=int, default=0)
    parser.add_argument("--split-strategy", choices=["random", "time"], default="random")

    parser.add_argument("--run-name", default="")
    parser.add_argument("--out-csv", default="tabtransformer_runs.csv")

    args = parser.parse_args()
    t0 = time.time()

    payload = torch.load(args.data, map_location="cpu")
    feature_set = payload.get("feature_set", "")
    n = int(payload["X_num"].shape[0])

    train_idx, val_idx, test_idx = make_splits(
        payload=payload,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        strategy=args.split_strategy,
    )

    # Optional train-cap for quick experiments (val/test stay full).
    train_max_samples = int(args.train_max_samples) if args.train_max_samples else 0
    if train_max_samples > 0 and train_max_samples < len(train_idx):
        rng = random.Random(args.seed + 9001)
        train_idx = rng.sample(train_idx, train_max_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_hidden_dims = _parse_int_list(args.mlp_hidden_dims)

    metrics = train_and_eval(
        payload=payload,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        seed=args.seed,
        clip_grad=args.clip_grad,
        val_max_samples=(int(args.val_max_samples) if args.val_max_samples else None),
        device=device,
        compute_test=True,
    )
    seconds = time.time() - t0

    run_name = args.run_name.strip() or f"tabtransformer_{args.split_strategy}_seed{args.seed}"

    row = {
        "source": "train_tabtransformer",
        "stage": "single_run",
        "run_name": run_name,
        "data": args.data,
        "feature_set": feature_set,
        "seed": args.seed,
        "split_seed": args.seed,
        "split_strategy": args.split_strategy,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "train_max_samples": ("" if train_max_samples <= 0 else train_max_samples),
        "n_train_full": len(make_splits(payload=payload, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed, strategy=args.split_strategy)[0]),
        "n_train_used": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "ff_mult": args.ff_mult,
        "dropout": args.dropout,
        "clip_grad": args.clip_grad,
        "val_max_samples": args.val_max_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mlp_hidden_dims": ",".join(str(x) for x in mlp_hidden_dims),
        "model": "tabtransformer",
        "seconds": round(seconds, 2),
        "device": device.type,
        **metrics,
    }

    out_path = Path(args.out_csv)
    append_row(out_path, row)
    print(f"Wrote CSV row: {out_path}")
    print(
        f"Done: split={args.split_strategy} n={n} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        f"  test_rmse={metrics['test_rmse']:.4f}  test_acc={metrics['test_acc_pct']:.2f}%  sec={seconds:.1f}"
    )


if __name__ == "__main__":
    main()

