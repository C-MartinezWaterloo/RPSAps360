#!/usr/bin/env python3
"""
Deep + long ANN sweep (TEST-FOCUSED).

You asked for:
  - Deeper networks
  - Lots of training (takes longer)
  - Focus on TEST RMSE (but still report train/val metrics)
  - All trials saved into one spreadsheet

Important note (statistics):
  - This script evaluates the TEST set for every configuration and ranks by
    test RMSE. That means the test set is being used like a validation set.
  - If you want a truly unbiased final number, keep a separate holdout set
    that this script never sees.

Outputs:
  - Writes `sweep_results_deep_test.csv`
  - Prints the top configs by test RMSE
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from typing import Any

import torch

from train_ann import _make_splits, train_and_eval


def _fmt_dims(hidden_dims: list[int]) -> str:
    return ",".join(str(x) for x in hidden_dims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors.pt")
    parser.add_argument("--out-csv", default="sweep_results_deep_test.csv")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs per run (best val epoch chosen within this)")
    parser.add_argument("--max-runs", type=int, default=80, help="How many configs to try (sampled from a larger grid)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If the output CSV already exists, append new runs after the existing ones.",
    )
    args = parser.parse_args()

    payload = torch.load(args.data, map_location="cpu")
    n = payload["X_num"].shape[0]

    # Fixed split so all configs are comparable.
    train_idx, val_idx, test_idx = _make_splits(n, 0.70, 0.15, 0.15, args.seed)

    # ---------------------------------------------------------------------
    # Config space (deeper than before). We build a grid and then sample.
    # ---------------------------------------------------------------------
    hidden_spaces = [
        ("deep3", [512, 256, 128]),
        ("deep4", [512, 256, 128, 64]),
        ("deep5", [1024, 512, 256, 128, 64]),
        ("deep6", [1024, 512, 512, 256, 128, 64]),
        ("deep6b", [2048, 1024, 512, 256, 128, 64]),
        ("deep7", [2048, 1024, 512, 512, 256, 128, 64]),
    ]

    # Smaller batch sizes = more gradient steps = slower, but sometimes better.
    batch_sizes = [512, 1024, 2048, 4096]

    # Learning rates to try.
    lrs = [0.0005, 0.001, 0.002, 0.003]

    # Regularization knobs.
    dropouts = [0.0, 0.05, 0.1]
    weight_decays = [0.0, 1e-4, 1e-3]

    # Embedding dimension cap (categoricals -> embeddings).
    embed_dim_caps = [32, 64]

    candidates: list[dict[str, Any]] = []
    for name, hidden_dims in hidden_spaces:
        for bs in batch_sizes:
            for lr in lrs:
                for do in dropouts:
                    for wd in weight_decays:
                        for emb_cap in embed_dim_caps:
                            candidates.append(
                                {
                                    "name": name,
                                    "hidden_dims": hidden_dims,
                                    "batch_size": bs,
                                    "lr": lr,
                                    "dropout": do,
                                    "weight_decay": wd,
                                    "embed_dim_cap": emb_cap,
                                }
                            )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    runs = candidates[: max(1, min(args.max_runs, len(candidates)))]

    # If resuming, skip the configs we already wrote to disk.
    start_run = 1
    if args.resume:
        try:
            with open(args.out_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                already = sum(1 for _ in reader)
            start_run = already + 1
        except FileNotFoundError:
            start_run = 1

    if start_run > len(runs):
        raise ValueError(f"--resume: output already has {start_run-1} rows, but only {len(runs)} runs requested.")

    print(
        f"Deep test-focused sweep: runs={len(runs)} / candidates={len(candidates)} "
        f"(epochs={args.epochs}, seed={args.seed}, split=70/15/15)"
    )

    # We write rows to CSV as we go (so if a long run is interrupted you still have results).
    fieldnames = [
        "source",
        "run",
        "seed",
        "split_seed",
        "epochs",
        "name",
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
        "seconds",
    ]

    write_header = not args.resume or start_run == 1
    mode = "w" if write_header else "a"
    with open(args.out_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for i, cfg in enumerate(runs[start_run - 1 :], start=start_run):
            start = time.time()
            metrics = train_and_eval(
                payload=payload,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                batch_size=int(cfg["batch_size"]),
                epochs=int(args.epochs),
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
                hidden_dims=list(cfg["hidden_dims"]),
                dropout=float(cfg["dropout"]),
                embed_dim_cap=int(cfg["embed_dim_cap"]),
                seed=args.seed,
                compute_test=True,  # <-- test for every run (test-focused)
            )
            elapsed_s = time.time() - start

            row = {
                "source": "sweep_deep_test",
                "run": i,
                "seed": args.seed,
                "split_seed": args.seed,
                "epochs": int(args.epochs),
                "name": cfg["name"],
                "hidden_dims": _fmt_dims(cfg["hidden_dims"]),
                "batch_size": int(cfg["batch_size"]),
                "lr": float(cfg["lr"]),
                "dropout": float(cfg["dropout"]),
                "weight_decay": float(cfg["weight_decay"]),
                "embed_dim_cap": int(cfg["embed_dim_cap"]),
                "n_params": metrics["n_params"],
                "best_val_epoch": metrics["best_val_epoch"],
                "train_mse": metrics["train_mse"],
                "val_mse": metrics["val_mse"],
                "test_mse": metrics["test_mse"],
                "train_rmse": metrics["train_rmse"],
                "val_rmse": metrics["val_rmse"],
                "test_rmse": metrics["test_rmse"],
                "seconds": round(elapsed_s, 2),
            }

            w.writerow(row)
            f.flush()

            print(
                f"[{i:03d}/{len(runs)}] {cfg['name']:<6} hidden={row['hidden_dims']:<24} "
                f"bs={row['batch_size']:<4} lr={row['lr']:<6} do={row['dropout']:<4} wd={row['weight_decay']:<6} "
                f"test_rmse={float(row['test_rmse']):.4f}  val_rmse={float(row['val_rmse']):.4f}  ({row['seconds']}s)"
            )

    # Load results back from CSV (so top-k works even if we resumed).
    results: list[dict[str, Any]] = []
    with open(args.out_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(dict(row))

    results_sorted = sorted(results, key=lambda r: float(r["test_rmse"]))
    print("\nTop configs by TEST RMSE (log1p price):")
    for r in results_sorted[: args.top_k]:
        print(
            f"  test_rmse={float(r['test_rmse']):.4f}  val_rmse={float(r['val_rmse']):.4f}  "
            f"name={r['name']} hidden={r['hidden_dims']} bs={r['batch_size']} lr={r['lr']} do={r['dropout']} wd={r['weight_decay']} emb_cap={r['embed_dim_cap']}"
        )

    print("\nSaved results:", args.out_csv)


if __name__ == "__main__":
    main()
