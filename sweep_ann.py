#!/usr/bin/env python3
"""Exhaustive ANN hyperparameter sweep with clear accuracy reporting."""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path


def _fmt_dims(hidden_dims: list[int]) -> str:
    return ",".join(str(x) for x in hidden_dims)


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_hidden_spaces(value: str) -> list[list[int]]:
    spaces: list[list[int]] = []
    for block in value.split(";"):
        block = block.strip()
        if not block:
            continue
        dims = [int(v.strip()) for v in block.split("-") if v.strip()]
        if dims:
            spaces.append(dims)
    if not spaces:
        raise ValueError("No hidden layer definitions were parsed.")
    return spaces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors.pt")
    parser.add_argument("--out-csv", default="sweep_results.csv", help="Single output file with all hyperparameter runs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)

    parser.add_argument("--batch-sizes", default="1024,2048,4096,8192")
    parser.add_argument("--lrs", default="0.0005,0.001,0.002,0.003")
    parser.add_argument("--dropouts", default="0.0,0.05,0.1,0.2")
    parser.add_argument("--weight-decays", default="0.0,0.0001,0.001")
    parser.add_argument("--embed-dim-caps", default="32,64")
    parser.add_argument("--hidden-spaces", default="128-64;256-128;512-256;512-256-128;1024-512-256")
    parser.add_argument("--limit-runs", type=int, default=0, help="0 means run all combinations (exhaustive)")
    parser.add_argument("--reset-results", action="store_true", help="Delete out-csv before running")
    args = parser.parse_args()

    if args.reset_results:
        out = Path(args.out_csv)
        if out.exists():
            out.unlink()

    import torch
    from train_ann import _make_splits, train_and_eval

    payload = torch.load(args.data, map_location="cpu")
    n = payload["X_num"].shape[0]
    train_idx, val_idx, test_idx = _make_splits(n, args.train_frac, args.val_frac, args.test_frac, args.seed)

    hidden_spaces = _parse_hidden_spaces(args.hidden_spaces)
    batch_sizes = _parse_int_list(args.batch_sizes)
    lrs = _parse_float_list(args.lrs)
    dropouts = _parse_float_list(args.dropouts)
    weight_decays = _parse_float_list(args.weight_decays)
    embed_dim_caps = _parse_int_list(args.embed_dim_caps)

    configs = list(itertools.product(hidden_spaces, batch_sizes, lrs, dropouts, weight_decays, embed_dim_caps))
    if args.limit_runs > 0:
        configs = configs[: args.limit_runs]

    fieldnames = [
        "run",
        "seed",
        "train_frac",
        "val_frac",
        "test_frac",
        "epochs",
        "hidden_dims",
        "batch_size",
        "lr",
        "dropout",
        "weight_decay",
        "embed_dim_cap",
        "n_params",
        "best_val_epoch",
        "train_mse",
        "val_mse",
        "test_mse",
        "train_rmse",
        "val_rmse",
        "test_rmse",
        "train_accuracy_pct",
        "val_accuracy_pct",
        "test_accuracy_pct",
        "seconds",
    ]

    rows: list[dict] = []
    print(f"Running {len(configs)} exhaustive hyperparameter runs.")

    for i, (hidden_dims, bs, lr, do, wd, emb_cap) in enumerate(configs, start=1):
        started = time.time()
        metrics = train_and_eval(
            payload=payload,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=int(bs),
            epochs=args.epochs,
            lr=float(lr),
            weight_decay=float(wd),
            hidden_dims=list(hidden_dims),
            dropout=float(do),
            embed_dim_cap=int(emb_cap),
            seed=args.seed,
            compute_test=True,
        )
        elapsed = round(time.time() - started, 2)

        row = {
            "run": i,
            "seed": args.seed,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "epochs": args.epochs,
            "hidden_dims": _fmt_dims(list(hidden_dims)),
            "batch_size": int(bs),
            "lr": float(lr),
            "dropout": float(do),
            "weight_decay": float(wd),
            "embed_dim_cap": int(emb_cap),
            "n_params": metrics["n_params"],
            "best_val_epoch": metrics["best_val_epoch"],
            "train_mse": metrics["train_mse"],
            "val_mse": metrics["val_mse"],
            "test_mse": metrics["test_mse"],
            "train_rmse": metrics["train_rmse"],
            "val_rmse": metrics["val_rmse"],
            "test_rmse": metrics["test_rmse"],
            "train_accuracy_pct": metrics["train_accuracy_pct"],
            "val_accuracy_pct": metrics["val_accuracy_pct"],
            "test_accuracy_pct": metrics["test_accuracy_pct"],
            "seconds": elapsed,
        }
        rows.append(row)

        print(
            f"[{i:04d}/{len(configs)}] hidden={row['hidden_dims']:<16} bs={row['batch_size']:<5} "
            f"lr={row['lr']:<7} do={row['dropout']:<4} wd={row['weight_decay']:<7} "
            f"acc% train/val/test={row['train_accuracy_pct']:.2f}/{row['val_accuracy_pct']:.2f}/{row['test_accuracy_pct']:.2f}"
        )

    rows_sorted = sorted(rows, key=lambda r: float(r["val_accuracy_pct"]), reverse=True)
    best = rows_sorted[0] if rows_sorted else None

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved results: {args.out_csv}")
    if best:
        print("Best by validation accuracy (%):")
        print(
            f"  val={best['val_accuracy_pct']:.2f}% test={best['test_accuracy_pct']:.2f}% "
            f"hidden={best['hidden_dims']} bs={best['batch_size']} lr={best['lr']} do={best['dropout']} wd={best['weight_decay']}"
        )


if __name__ == "__main__":
    main()
