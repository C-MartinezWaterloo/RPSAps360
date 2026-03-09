#!/usr/bin/env python3
"""
Permutation feature importance for tabular models in this repo.

We compute importance on a validation subset by measuring the change in:
  - val_mse (log1p space)
  - val_acc_pct (MdAPE-derived accuracy % in price space)

when a single feature column is permuted across rows.

Outputs:
  - Writes `feature_importance_results.csv` (gitignored) for easy Excel viewing.
  - Intended to support explanations in REPORT.md.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from hedonic_baseline import HedonicDataset, HedonicLinear
from train_ann import PriceANN, TensorDataset, make_splits


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Expected a comma-separated list of integers.")
    return out


def _metrics_from_preds(y_log1p: torch.Tensor, pred_log1p: torch.Tensor) -> dict[str, float]:
    mse = torch.mean((pred_log1p.view(-1) - y_log1p.view(-1)) ** 2).item()
    rmse = float(mse**0.5)
    eps = 1e-8
    y_true = torch.expm1(y_log1p.view(-1))
    y_pred = torch.expm1(pred_log1p.view(-1)).clamp_min(0.0)
    ape = (y_pred - y_true).abs() / y_true.clamp_min(eps)
    mdape = torch.median(ape).item()
    acc_pct = max(0.0, 100.0 * (1.0 - float(mdape)))
    return {"mse": float(mse), "rmse": float(rmse), "mdape": float(mdape), "acc_pct": float(acc_pct)}


@torch.no_grad()
def _eval_model_ann(model: nn.Module, x_num: torch.Tensor, x_cat: torch.Tensor, y_log1p: torch.Tensor) -> dict[str, float]:
    model.eval()
    pred = model(x_num, x_cat)
    return _metrics_from_preds(y_log1p, pred)


@torch.no_grad()
def _eval_model_hedonic(model: nn.Module, x_num: torch.Tensor, x_cat: torch.Tensor, y_log1p: torch.Tensor) -> dict[str, float]:
    model.eval()
    pred = model(x_num, x_cat)
    return _metrics_from_preds(y_log1p, pred)


def _train_ann(
    *,
    payload: dict,
    train_idx: list[int],
    val_idx: list[int],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dims: list[int],
    dropout: float,
    embed_dim_cap: int,
    device: torch.device,
) -> nn.Module:
    ds = TensorDataset(payload)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)

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

    best_val = float("inf")
    best_state: dict | None = None

    for _epoch in range(1, epochs + 1):
        model.train()
        for x_num, x_cat, y in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x_num.to(device), x_cat.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            opt.step()

        model.eval()
        val_mse = 0.0
        n = 0
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                pred = model(x_num.to(device), x_cat.to(device))
                diff = pred - y.to(device)
                val_mse += float((diff * diff).sum().item())
                n += int(y.shape[0])
        val_mse = val_mse / max(1, n)
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _train_hedonic(
    *,
    payload: dict,
    train_idx: list[int],
    val_idx: list[int],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> nn.Module:
    ds = HedonicDataset(payload)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)

    cat_sizes = [len(payload["cat_maps"][col]) for col in payload["cat_cols"]]
    model = HedonicLinear(n_num=payload["X_num"].shape[1], cat_sizes=cat_sizes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict | None = None

    for _epoch in range(1, epochs + 1):
        model.train()
        for x_num, x_cat, y_log, _y_raw in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x_num.to(device), x_cat.to(device))
            loss = loss_fn(pred, y_log.to(device))
            loss.backward()
            opt.step()

        model.eval()
        val_mse = 0.0
        n = 0
        with torch.no_grad():
            for x_num, x_cat, y_log, _y_raw in val_loader:
                pred = model(x_num.to(device), x_cat.to(device))
                diff = pred - y_log.to(device)
                val_mse += float((diff * diff).sum().item())
                n += int(y_log.shape[0])
        val_mse = val_mse / max(1, n)
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors_full.pt")
    parser.add_argument("--out-csv", default="feature_importance_results.csv")
    parser.add_argument("--split-strategy", choices=["random", "time"], default="random")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-max-samples", type=int, default=20_000)
    parser.add_argument("--perm-seed", type=int, default=0, help="Seed for permutation randomness.")

    parser.add_argument("--ann-seed", type=int, default=0)
    parser.add_argument("--ann-hidden-dims", default="512,256,128,64")
    parser.add_argument("--ann-dropout", type=float, default=0.1)
    parser.add_argument("--ann-embed-dim-cap", type=int, default=64)
    parser.add_argument("--ann-epochs", type=int, default=30)
    parser.add_argument("--ann-batch-size", type=int, default=512)
    parser.add_argument("--ann-lr", type=float, default=0.004)
    parser.add_argument("--ann-weight-decay", type=float, default=0.001)

    parser.add_argument("--hedonic-seed", type=int, default=0)
    parser.add_argument("--hedonic-epochs", type=int, default=20)
    parser.add_argument("--hedonic-batch-size", type=int, default=8192)
    parser.add_argument("--hedonic-lr", type=float, default=0.01)
    parser.add_argument("--hedonic-weight-decay", type=float, default=0.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu")

    train_idx, val_idx, _test_idx = make_splits(
        payload=payload, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=args.split_seed, strategy=args.split_strategy
    )

    if args.val_max_samples and args.val_max_samples < len(val_idx):
        val_idx = val_idx[: args.val_max_samples]

    # Build val tensors in one go (faster for repeated passes).
    x_num_val = payload["X_num"][val_idx].to(device)
    x_cat_val = payload["X_cat"][val_idx].to(device)
    y_val = payload["y_log1p"][val_idx].to(device)

    perm_rng = np.random.default_rng(args.perm_seed)

    started = time.time()

    ann_hidden_dims = _parse_int_list(args.ann_hidden_dims)
    ann = _train_ann(
        payload=payload,
        train_idx=train_idx,
        val_idx=val_idx,
        seed=args.ann_seed,
        epochs=args.ann_epochs,
        batch_size=args.ann_batch_size,
        lr=args.ann_lr,
        weight_decay=args.ann_weight_decay,
        hidden_dims=ann_hidden_dims,
        dropout=args.ann_dropout,
        embed_dim_cap=args.ann_embed_dim_cap,
        device=device,
    )
    hed = _train_hedonic(
        payload=payload,
        train_idx=train_idx,
        val_idx=val_idx,
        seed=args.hedonic_seed,
        epochs=args.hedonic_epochs,
        batch_size=args.hedonic_batch_size,
        lr=args.hedonic_lr,
        weight_decay=args.hedonic_weight_decay,
        device=device,
    )

    base_ann = _eval_model_ann(ann, x_num_val, x_cat_val, y_val)
    base_hed = _eval_model_hedonic(hed, x_num_val, x_cat_val, y_val)

    numeric_cols = list(payload.get("numeric_cols", []))
    cat_cols = list(payload.get("cat_cols", []))

    rows: list[dict] = []

    def _permute_col(x: torch.Tensor, col_idx: int) -> torch.Tensor:
        x2 = x.clone()
        perm = perm_rng.permutation(x2.shape[0])
        x2[:, col_idx] = x2[perm, col_idx]
        return x2

    # ANN permutation importance.
    for j, col in enumerate(numeric_cols):
        x_num_p = _permute_col(x_num_val, j)
        m = _eval_model_ann(ann, x_num_p, x_cat_val, y_val)
        rows.append(
            {
                "model": "ann_mlp",
                "split_strategy": args.split_strategy,
                "split_seed": args.split_seed,
                "feature_type": "numeric",
                "feature_name": col,
                "baseline_val_mse": base_ann["mse"],
                "baseline_val_acc_pct": base_ann["acc_pct"],
                "delta_val_mse": float(m["mse"] - base_ann["mse"]),
                "delta_val_acc_pct": float(m["acc_pct"] - base_ann["acc_pct"]),
            }
        )
    for j, col in enumerate(cat_cols):
        x_cat_p = _permute_col(x_cat_val, j)
        m = _eval_model_ann(ann, x_num_val, x_cat_p, y_val)
        rows.append(
            {
                "model": "ann_mlp",
                "split_strategy": args.split_strategy,
                "split_seed": args.split_seed,
                "feature_type": "categorical",
                "feature_name": col,
                "baseline_val_mse": base_ann["mse"],
                "baseline_val_acc_pct": base_ann["acc_pct"],
                "delta_val_mse": float(m["mse"] - base_ann["mse"]),
                "delta_val_acc_pct": float(m["acc_pct"] - base_ann["acc_pct"]),
            }
        )

    # Hedonic permutation importance.
    for j, col in enumerate(numeric_cols):
        x_num_p = _permute_col(x_num_val, j)
        m = _eval_model_hedonic(hed, x_num_p, x_cat_val, y_val)
        rows.append(
            {
                "model": "hedonic_linear_fixed_effects",
                "split_strategy": args.split_strategy,
                "split_seed": args.split_seed,
                "feature_type": "numeric",
                "feature_name": col,
                "baseline_val_mse": base_hed["mse"],
                "baseline_val_acc_pct": base_hed["acc_pct"],
                "delta_val_mse": float(m["mse"] - base_hed["mse"]),
                "delta_val_acc_pct": float(m["acc_pct"] - base_hed["acc_pct"]),
            }
        )
    for j, col in enumerate(cat_cols):
        x_cat_p = _permute_col(x_cat_val, j)
        m = _eval_model_hedonic(hed, x_num_val, x_cat_p, y_val)
        rows.append(
            {
                "model": "hedonic_linear_fixed_effects",
                "split_strategy": args.split_strategy,
                "split_seed": args.split_seed,
                "feature_type": "categorical",
                "feature_name": col,
                "baseline_val_mse": base_hed["mse"],
                "baseline_val_acc_pct": base_hed["acc_pct"],
                "delta_val_mse": float(m["mse"] - base_hed["mse"]),
                "delta_val_acc_pct": float(m["acc_pct"] - base_hed["acc_pct"]),
            }
        )

    out_path = Path(args.out_csv)
    fieldnames = [
        "model",
        "split_strategy",
        "split_seed",
        "feature_type",
        "feature_name",
        "baseline_val_mse",
        "baseline_val_acc_pct",
        "delta_val_mse",
        "delta_val_acc_pct",
    ]
    _write_rows(out_path, fieldnames, rows)

    elapsed = time.time() - started
    print(f"Wrote: {out_path}  rows={len(rows)}  elapsed={elapsed:.1f}s")

    def _top(model: str, k: int = 8):
        subset = [r for r in rows if r["model"] == model]
        subset.sort(key=lambda r: float(r["delta_val_mse"]), reverse=True)
        return subset[:k]

    print("\nTop features by Δval_mse (ANN):")
    for r in _top("ann_mlp", k=10):
        print(f"  {r['feature_type']:<11} {r['feature_name']:<22}  Δmse={float(r['delta_val_mse']):.6f}  Δacc={float(r['delta_val_acc_pct']):+.2f}%")

    print("\nTop features by Δval_mse (Hedonic):")
    for r in _top("hedonic_linear_fixed_effects", k=10):
        print(f"  {r['feature_type']:<11} {r['feature_name']:<22}  Δmse={float(r['delta_val_mse']):.6f}  Δacc={float(r['delta_val_acc_pct']):+.2f}%")


if __name__ == "__main__":
    main()

