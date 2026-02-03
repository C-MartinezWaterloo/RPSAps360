#!/usr/bin/env python3
"""
Export all experiment results into a single Excel workbook.

This creates `ann_results.xlsx` with ONE sheet ("results") that contains:
  - manual_runs         (printed earlier in chat; recorded below)
  - sweep_dirty         (older sweep that computed test for every config)
  - sweep_clean         (newer sweep that is test-clean; test evaluated once)

No pandas is used (pandas is broken in this environment); we use:
  - csv (stdlib) to read the sweep CSV
  - openpyxl (pure python) to write .xlsx
"""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font


def _write_sheet(ws, rows: list[dict]) -> None:
    """Write a list of dict rows to an openpyxl worksheet."""
    if not rows:
        ws.append(["(no rows)"])
        return

    # Use the union of keys (stable order: keys of first row, then any extras).
    header: list[str] = list(rows[0].keys())
    extra_keys = sorted({k for r in rows for k in r.keys()} - set(header))
    header.extend(extra_keys)

    # Header row
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    # Data rows
    for r in rows:
        ws.append([r.get(k) for k in header])

    ws.freeze_panes = "A2"


def _maybe_number(x: str):
    """
    Convert a CSV string to int/float when it looks like a number.

    This makes the Excel output nicer (real numbers you can sort/filter).
    """

    if x is None:
        return ""
    s = str(x).strip()
    if s == "":
        return ""

    # Integer?
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        try:
            return int(s)
        except ValueError:
            return s

    # Float?
    try:
        return float(s)
    except ValueError:
        return s


def main() -> None:
    root = Path(__file__).resolve().parent

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

    # 2) Old sweep (NOT test-clean): computed test metrics for every config.
    sweep_dirty_csv = root / "sweep_results.csv"
    if sweep_dirty_csv.exists():
        with sweep_dirty_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_clean = {k: _maybe_number(v) for k, v in row.items()}
                row_clean["source"] = "sweep_dirty"
                row_clean.setdefault("stage", "sweep_dirty")
                rows_all.append(row_clean)

    # 3) New clean sweep: test evaluated once (stage=final_test only).
    sweep_clean_csv = root / "sweep_results_clean.csv"
    if sweep_clean_csv.exists():
        with sweep_clean_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_clean = {k: _maybe_number(v) for k, v in row.items()}
                row_clean["source"] = "sweep_clean"
                rows_all.append(row_clean)

    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    _write_sheet(ws, rows_all)

    out_path = root / "ann_results.xlsx"
    wb.save(out_path)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
