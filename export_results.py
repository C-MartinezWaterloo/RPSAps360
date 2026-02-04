#!/usr/bin/env python3
"""
Export all experiment results into a single CSV.

Goal (per your request):
  - Keep ONE canonical results file for GitHub: `results_all.csv`
  - Merge everything we ran so far:
      * manual runs (hard-coded below)
      * single ANN runs (ann_runs.csv)
      * sweeps (sweep_results*.csv)
      * hedonic baseline (hedonic_results.csv)

We avoid pandas (it's broken in this environment) and use only stdlib csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

def _load_csv_rows(path: Path) -> list[dict]:
    """Load rows from a CSV file (as dicts)."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def main() -> None:
    root = Path(__file__).resolve().parent
    out_csv = root / "results_all.csv"

    # These are the specific configs/metrics we ran earlier (before the big sweep)
    # using `train_ann.py` with a 70/15/15 split and seed=42.
    manual_runs: list[dict] = [
        {
            "run": 1,
            "notes": "3 epochs baseline",
            "epochs": 3,
            "batch_size": 4096,
            "seed": 42,
            "hidden_dims": "256,128",
            "dropout": 0.1,
            "embed_dim_cap": 64,
            "lr": 0.001,
            "weight_decay": 0.0,
            "best_val_epoch": 3,
            "best_val_mse": 0.237703,
            "train_mse": 0.229767,
            "val_mse": 0.237703,
            "test_mse": 0.241721,
            "train_rmse": 0.479341,
            "val_rmse": 0.487548,
            "test_rmse": 0.491651,
        },
        {
            "run": 2,
            "notes": "3 epochs, low lr",
            "epochs": 3,
            "batch_size": 4096,
            "seed": 42,
            "hidden_dims": "256,128",
            "dropout": 0.1,
            "embed_dim_cap": 64,
            "lr": 0.0003,
            "weight_decay": 0.0,
            "best_val_epoch": 3,
            "best_val_mse": 0.655213,
            "train_mse": 0.655124,
            "val_mse": 0.655213,
            "test_mse": 0.757485,
            "train_rmse": 0.809397,
            "val_rmse": 0.809452,
            "test_rmse": 0.870336,
        },
        {
            "run": 3,
            "notes": "5 epochs baseline",
            "epochs": 5,
            "batch_size": 4096,
            "seed": 42,
            "hidden_dims": "256,128",
            "dropout": 0.1,
            "embed_dim_cap": 64,
            "lr": 0.001,
            "weight_decay": 0.0,
            "best_val_epoch": 5,
            "best_val_mse": 0.181576,
            "train_mse": 0.175140,
            "val_mse": 0.181576,
            "test_mse": 0.183442,
            "train_rmse": 0.418498,
            "val_rmse": 0.426117,
            "test_rmse": 0.428302,
        },
        {
            "run": 4,
            "notes": "5 epochs, lr=5e-4",
            "epochs": 5,
            "batch_size": 4096,
            "seed": 42,
            "hidden_dims": "256,128",
            "dropout": 0.1,
            "embed_dim_cap": 64,
            "lr": 0.0005,
            "weight_decay": 0.0,
            "best_val_epoch": 5,
            "best_val_mse": 0.245241,
            "train_mse": 0.237944,
            "val_mse": 0.245241,
            "test_mse": 0.253632,
            "train_rmse": 0.487795,
            "val_rmse": 0.495218,
            "test_rmse": 0.503619,
        },
        {
            "run": 5,
            "notes": "5 epochs, lr=0.002",
            "epochs": 5,
            "batch_size": 4096,
            "seed": 42,
            "hidden_dims": "256,128",
            "dropout": 0.1,
            "embed_dim_cap": 64,
            "lr": 0.002,
            "weight_decay": 0.0,
            "best_val_epoch": 5,
            "best_val_mse": 0.150653,
            "train_mse": 0.145335,
            "val_mse": 0.150653,
            "test_mse": 0.152204,
            "train_rmse": 0.381229,
            "val_rmse": 0.388141,
            "test_rmse": 0.390133,
        },
        {
            "run": 6,
            "notes": "smaller model",
            "epochs": 5,
            "batch_size": 4096,
            "seed": 42,
            "hidden_dims": "128,64",
            "dropout": 0.1,
            "embed_dim_cap": 64,
            "lr": 0.002,
            "weight_decay": 0.0,
            "best_val_epoch": 5,
            "best_val_mse": 0.167460,
            "train_mse": 0.163305,
            "val_mse": 0.167460,
            "test_mse": 0.168680,
            "train_rmse": 0.404110,
            "val_rmse": 0.409218,
            "test_rmse": 0.410707,
        },
        {
            "run": 7,
            "notes": "no dropout",
            "epochs": 5,
            "batch_size": 4096,
            "seed": 42,
            "hidden_dims": "256,128",
            "dropout": 0.0,
            "embed_dim_cap": 64,
            "lr": 0.002,
            "weight_decay": 0.0,
            "best_val_epoch": 5,
            "best_val_mse": 0.143858,
            "train_mse": 0.138823,
            "val_mse": 0.143858,
            "test_mse": 0.145424,
            "train_rmse": 0.372589,
            "val_rmse": 0.379287,
            "test_rmse": 0.381345,
        },
        {
            "run": 8,
            "notes": "no dropout + weight decay",
            "epochs": 5,
            "batch_size": 4096,
            "seed": 42,
            "hidden_dims": "256,128",
            "dropout": 0.0,
            "embed_dim_cap": 64,
            "lr": 0.002,
            "weight_decay": 0.01,
            "best_val_epoch": 5,
            "best_val_mse": 0.143954,
            "train_mse": 0.138898,
            "val_mse": 0.143954,
            "test_mse": 0.145555,
            "train_rmse": 0.372690,
            "val_rmse": 0.379413,
            "test_rmse": 0.381517,
        },
    ]

    rows_all: list[dict] = []

    # 1) Manual runs (already numeric).
    for r in manual_runs:
        rows_all.append({"source": "manual_runs", "stage": "manual", **r})

    # 2) Logged single ANN runs.
    for row in _load_csv_rows(root / "ann_runs.csv"):
        row.setdefault("source", "train_ann")
        row.setdefault("stage", "single_run")
        rows_all.append(row)

    # 3) Sweep results (older / not test-clean).
    for row in _load_csv_rows(root / "sweep_results.csv"):
        row.setdefault("source", "sweep_dirty")
        row.setdefault("stage", "sweep_dirty")
        rows_all.append(row)

    # 4) Sweep results (test-clean).
    for row in _load_csv_rows(root / "sweep_results_clean.csv"):
        row.setdefault("source", "sweep_clean")
        rows_all.append(row)

    # 5) Deep sweep (test-focused).
    for row in _load_csv_rows(root / "sweep_results_deep_test.csv"):
        row.setdefault("source", "sweep_deep_test")
        row.setdefault("stage", "sweep_deep_test")
        rows_all.append(row)

    # 6) Hedonic baseline.
    for row in _load_csv_rows(root / "hedonic_results.csv"):
        row.setdefault("source", "hedonic")
        row.setdefault("stage", "hedonic_baseline")
        rows_all.append(row)

    # Build a stable header: common keys first, then the rest alphabetically.
    preferred = [
        "source",
        "stage",
        "run",
        "run_name",
        "name",
        "data",
        "feature_set",
        "seed",
        "split_seed",
        "train_frac",
        "val_frac",
        "test_frac",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dropout",
        "embed_dim_cap",
        "hidden_dims",
        "n_params",
        "best_val_epoch",
        "best_val_mse",
        "train_mse",
        "val_mse",
        "test_mse",
        "train_rmse",
        "val_rmse",
        "test_rmse",
        "test_price_rmse",
        "seconds",
    ]
    all_keys = {k for r in rows_all for k in r.keys()}
    header = [k for k in preferred if k in all_keys]
    header += sorted(all_keys - set(header))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows_all:
            # Ensure missing keys become empty strings in output.
            w.writerow({k: r.get(k, "") for k in header})

    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
