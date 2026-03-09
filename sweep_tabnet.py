#!/usr/bin/env python3
"""
Run a randomized sweep for TabNet.

Outputs:
  - Writes a single sweep CSV (default: tabnet_sweep_results.csv; gitignored)
  - Merge into the canonical `results_all.csv` via `python export_results.py`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_ann import make_splits
from train_tabnet import train_and_eval


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _config_id(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def _write_row(path: Path, fieldnames: list[str], row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors_full.pt")
    parser.add_argument("--out-csv", default="tabnet_sweep_results.csv")
    parser.add_argument("--n-runs", type=int, default=200)

    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "time"],
        default="random",
        help="How to split data into train/val/test (default: random). 'time' sorts by TransactionYear/Quarter.",
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument(
        "--val-max-samples",
        type=int,
        default=20_000,
        help="Cap validation rows used for best-epoch selection (full val/test metrics still computed at the end). Use 0 for full val.",
    )

    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=50_000,
        help="Optional: cap training rows for sweep speed (val/test still full). Use 0 for full train split.",
    )

    parser.add_argument("--config-seed", type=int, default=0)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-results", action="store_true")
    args = parser.parse_args()

    if args.n_runs <= 0:
        raise SystemExit("--n-runs must be positive.")
    if args.train_max_samples < 0:
        raise SystemExit("--train-max-samples must be >= 0.")
    if args.val_max_samples < 0:
        raise SystemExit("--val-max-samples must be >= 0.")
    if args.candidate_offset < 0:
        raise SystemExit("--candidate-offset must be >= 0.")

    out_path = Path(args.out_csv)
    if args.reset_results and out_path.exists():
        out_path.unlink()

    existing_ids: set[str] = set()
    next_run = 1
    best_by_val = None
    best_by_test = None
    if args.resume and out_path.exists():
        existing = _load_csv_rows(out_path)
        for r in existing:
            cid = str(r.get("config_id", "")).strip()
            if cid:
                existing_ids.add(cid)
        run_vals = []
        for r in existing:
            try:
                run_vals.append(int(str(r.get("run", "")).strip()))
            except Exception:
                pass
        next_run = (max(run_vals) + 1) if run_vals else (len(existing) + 1)

        for r in existing:
            try:
                v = float(r.get("val_acc_pct", ""))
            except Exception:
                v = None
            if v is not None and (best_by_val is None or v > float(best_by_val.get("val_acc_pct", "-inf"))):
                best_by_val = r

            try:
                t = float(r.get("test_acc_pct", ""))
            except Exception:
                t = None
            if t is not None and (best_by_test is None or t > float(best_by_test.get("test_acc_pct", "-inf"))):
                best_by_test = r

        print(f"[resume] existing rows={len(existing)} unique_config_ids={len(existing_ids)} next_run={next_run}")

    payload = torch.load(args.data, map_location="cpu")
    n_total = int(payload["X_num"].shape[0])

    train_idx_full, val_idx, test_idx = make_splits(
        payload=payload,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.split_seed,
        strategy=args.split_strategy,
    )
    n_train_full = len(train_idx_full)
    n_val = len(val_idx)
    n_test = len(test_idx)

    if args.train_max_samples and args.train_max_samples < n_train_full:
        train_idx = train_idx_full[: args.train_max_samples]
    else:
        train_idx = train_idx_full
    n_train_used = len(train_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Search space (kept moderate for CPU).
    embed_dim_caps = [32, 64]
    cat_dropouts = [0.0, 0.05]
    # Starter sweep space (kept moderate for CPU so we can run lots of configs).
    n_ds = [16, 32, 64]
    n_as = [16, 32, 64]
    n_steps = [3, 4, 5]
    gammas = [1.3, 1.5, 1.7]
    n_shared = [1, 2]
    n_independent = [1, 2]
    dropouts = [0.0, 0.05, 0.1]
    lambda_sparses = [0.0, 1e-6, 1e-5, 1e-4]
    epochs = [15, 20]
    batch_sizes = [2048, 4096]
    lrs = [0.0005, 0.001, 0.002, 0.003, 0.005]
    weight_decays = [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]

    rng = np.random.default_rng(args.config_seed)

    fieldnames = [
        "source",
        "stage",
        "run",
        "config_id",
        "data",
        "feature_set",
        "seed",
        "split_seed",
        "split_strategy",
        "candidate_offset",
        "train_max_samples",
        "n_total",
        "n_train_full",
        "n_train_used",
        "n_val",
        "n_test",
        "epochs",
        "name",
        "embed_dim_cap",
        "cat_dropout",
        "n_d",
        "n_a",
        "n_steps",
        "gamma",
        "n_shared",
        "n_independent",
        "dropout",
        "lambda_sparse",
        "batch_size",
        "lr",
        "weight_decay",
        "n_params",
        "best_val_epoch",
        "best_val_mse",
        "train_mse",
        "val_mse",
        "test_mse",
        "train_rmse",
        "val_rmse",
        "test_rmse",
        "train_mdape",
        "val_mdape",
        "test_mdape",
        "train_acc_pct",
        "val_acc_pct",
        "test_acc_pct",
        "train_mse_last",
        "val_mse_last",
        "test_mse_last",
        "train_rmse_last",
        "val_rmse_last",
        "test_rmse_last",
        "train_mdape_last",
        "val_mdape_last",
        "test_mdape_last",
        "train_acc_pct_last",
        "val_acc_pct_last",
        "test_acc_pct_last",
        "seconds",
        "model",
    ]

    unique_seen: set[str] = set()
    generated: list[dict] = []

    def _sample_config() -> dict:
        return {
            "embed_dim_cap": int(rng.choice(embed_dim_caps)),
            "cat_dropout": float(rng.choice(cat_dropouts)),
            "n_d": int(rng.choice(n_ds)),
            "n_a": int(rng.choice(n_as)),
            "n_steps": int(rng.choice(n_steps)),
            "gamma": float(rng.choice(gammas)),
            "n_shared": int(rng.choice(n_shared)),
            "n_independent": int(rng.choice(n_independent)),
            "dropout": float(rng.choice(dropouts)),
            "lambda_sparse": float(rng.choice(lambda_sparses)),
            "epochs": int(rng.choice(epochs)),
            "batch_size": int(rng.choice(batch_sizes)),
            "lr": float(rng.choice(lrs)),
            "weight_decay": float(rng.choice(weight_decays)),
        }

    max_attempts = 600_000
    attempts = 0
    while len(generated) < args.n_runs and attempts < max_attempts:
        attempts += 1
        hp = _sample_config()
        config = {
            "model": "tabnet",
            "data": args.data,
            "split_seed": int(args.split_seed),
            "split_strategy": args.split_strategy,
            "train_max_samples": int(args.train_max_samples),
            **hp,
        }
        cid = _config_id(config)
        if cid in unique_seen:
            continue
        unique_seen.add(cid)
        if len(unique_seen) <= args.candidate_offset:
            continue
        if args.resume and cid in existing_ids:
            continue
        generated.append({"config_id": cid, **hp})

    if len(generated) < args.n_runs:
        print(f"[warn] Only generated {len(generated)}/{args.n_runs} unique configs after {attempts} attempts.")

    print("\nTabNet sweep:")
    print(f"  data: {args.data}  feature_set={payload.get('feature_set','')}")
    print(f"  split: {args.split_strategy} seed={args.split_seed}  train/val/test={args.train_frac}/{args.val_frac}/{args.test_frac}")
    print(f"  train_max_samples: {args.train_max_samples} (train_used={n_train_used}/{n_train_full})  val={n_val} test={n_test} total={n_total}")
    print(f"  val_max_samples (for epoch selection): {args.val_max_samples if args.val_max_samples else 'full'}")
    print(f"  device: {device.type}")
    print(f"  out_csv: {out_path}")
    print(f"  configs: target_runs={args.n_runs} generated={len(generated)} offset={args.candidate_offset} resume={args.resume}")

    for i, cfg in enumerate(generated, start=0):
        run_id = next_run + i
        start = time.time()

        metrics = train_and_eval(
            payload=payload,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            embed_dim_cap=cfg["embed_dim_cap"],
            cat_dropout=cfg["cat_dropout"],
            n_d=cfg["n_d"],
            n_a=cfg["n_a"],
            n_steps=cfg["n_steps"],
            gamma=cfg["gamma"],
            n_shared=cfg["n_shared"],
            n_independent=cfg["n_independent"],
            dropout=cfg["dropout"],
            lambda_sparse=cfg["lambda_sparse"],
            seed=args.split_seed,
            val_max_samples=None if args.val_max_samples == 0 else int(args.val_max_samples),
            device=device,
            compute_test=True,
        )

        seconds = time.time() - start
        name = (
            f"tabnet_ed{cfg['embed_dim_cap']}_cd{cfg['cat_dropout']}"
            f"_d{cfg['n_d']}_a{cfg['n_a']}_s{cfg['n_steps']}_g{cfg['gamma']}"
            f"_sh{cfg['n_shared']}_in{cfg['n_independent']}"
            f"_do{cfg['dropout']}_ls{cfg['lambda_sparse']}"
            f"_e{cfg['epochs']}_bs{cfg['batch_size']}_lr{cfg['lr']}_wd{cfg['weight_decay']}"
        )

        row = {
            "source": "sweep_tabnet",
            "stage": "tabnet",
            "run": int(run_id),
            "config_id": cfg["config_id"],
            "data": args.data,
            "feature_set": payload.get("feature_set", ""),
            "seed": int(args.split_seed),
            "split_seed": int(args.split_seed),
            "split_strategy": args.split_strategy,
            "candidate_offset": int(args.candidate_offset),
            "train_max_samples": int(args.train_max_samples),
            "n_total": int(n_total),
            "n_train_full": int(n_train_full),
            "n_train_used": int(n_train_used),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "epochs": int(cfg["epochs"]),
            "name": name,
            "embed_dim_cap": int(cfg["embed_dim_cap"]),
            "cat_dropout": float(cfg["cat_dropout"]),
            "n_d": int(cfg["n_d"]),
            "n_a": int(cfg["n_a"]),
            "n_steps": int(cfg["n_steps"]),
            "gamma": float(cfg["gamma"]),
            "n_shared": int(cfg["n_shared"]),
            "n_independent": int(cfg["n_independent"]),
            "dropout": float(cfg["dropout"]),
            "lambda_sparse": float(cfg["lambda_sparse"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
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

        _write_row(out_path, fieldnames, row)

        if best_by_val is None or float(row["val_acc_pct"]) > float(best_by_val["val_acc_pct"]):
            best_by_val = row
        if best_by_test is None or float(row["test_acc_pct"]) > float(best_by_test["test_acc_pct"]):
            best_by_test = row

        print(
            f"run {run_id:>5}  {name[:56]:<56}"
            f"  val_acc={float(row['val_acc_pct']):6.2f}%  test_acc={float(row['test_acc_pct']):6.2f}%"
            f"  (last: val={float(row['val_acc_pct_last']):6.2f}% test={float(row['test_acc_pct_last']):6.2f}%)"
            f"  sec={row['seconds']}"
        )

    if best_by_val:
        print("\nBest by validation accuracy (%):")
        print(f"  run={best_by_val['run']}  {best_by_val['name']}")
        print(f"  val_acc={float(best_by_val['val_acc_pct']):.2f}%  test_acc={float(best_by_val['test_acc_pct']):.2f}%")
        print(
            "  last_epoch:"
            f" val_acc={float(best_by_val['val_acc_pct_last']):.2f}%"
            f" test_acc={float(best_by_val['test_acc_pct_last']):.2f}%"
        )
        print(f"  out_csv: {out_path}")

    if best_by_test:
        print("\nBest by test accuracy (%):")
        print(f"  run={best_by_test['run']}  {best_by_test['name']}")
        print(f"  test_acc={float(best_by_test['test_acc_pct']):.2f}%  val_acc={float(best_by_test['val_acc_pct']):.2f}%")


if __name__ == "__main__":
    main()
