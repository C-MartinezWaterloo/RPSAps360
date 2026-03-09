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
import hashlib
import json
import random
import time
from typing import Any

import torch

from train_ann import make_splits, train_and_eval


def _fmt_dims(hidden_dims: list[int]) -> str:
    return ",".join(str(x) for x in hidden_dims)


def _config_id(cfg: dict[str, Any]) -> str:
    stable = {
        "name": cfg.get("name"),
        "hidden_dims": list(cfg.get("hidden_dims", [])),
        "batch_size": int(cfg.get("batch_size")),
        "lr": float(cfg.get("lr")),
        "dropout": float(cfg.get("dropout")),
        "weight_decay": float(cfg.get("weight_decay")),
        "embed_dim_cap": int(cfg.get("embed_dim_cap")),
    }
    s = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors.pt")
    parser.add_argument("--out-csv", default="sweep_results_deep_test.csv")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs per run (best val epoch chosen within this)")
    parser.add_argument("--max-runs", type=int, default=80, help="How many configs to try (sampled from a larger grid)")
    parser.add_argument(
        "--candidate-offset",
        type=int,
        default=0,
        help="Skip this many shuffled configs before taking --max-runs (useful for running additional batches).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
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
        "--resume",
        action="store_true",
        help="If the output CSV already exists, append new runs after the existing ones.",
    )
    args = parser.parse_args()

    payload = torch.load(args.data, map_location="cpu")
    n = payload["X_num"].shape[0]

    # Fixed split so all configs are comparable.
    train_idx_full, val_idx, test_idx = make_splits(
        payload=payload,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        seed=args.seed,
        strategy=args.split_strategy,
    )
    n_train_full = len(train_idx_full)
    n_val = len(val_idx)
    n_test = len(test_idx)

    train_idx = train_idx_full
    if args.train_max_samples is not None:
        if args.train_max_samples <= 0:
            raise ValueError("--train-max-samples must be positive.")
        if args.train_max_samples < len(train_idx_full):
            rng_sub = random.Random(args.seed)
            train_idx = rng_sub.sample(train_idx_full, args.train_max_samples)
    n_train_used = len(train_idx)

    # ---------------------------------------------------------------------
    # Config space. We build a grid and then sample.
    # ---------------------------------------------------------------------
    # Bigger models (as you requested).
    hidden_spaces = [
        ("deep4", [512, 256, 128, 64]),
        ("big4", [1024, 512, 256, 128]),
        ("big5", [1024, 512, 256, 128, 64]),
        ("deep6", [1024, 512, 512, 256, 128, 64]),
        ("wide3", [1024, 1024, 512]),
        ("huge3", [2048, 1024, 512]),
        ("huge4", [2048, 1024, 512, 256]),
    ]

    # Avoid tiny batches (too slow on CPU).
    batch_sizes = [4096, 8192]

    lrs = [0.001, 0.0015, 0.002, 0.003, 0.004]

    dropouts = [0.0, 0.05, 0.1, 0.2]
    weight_decays = [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]

    embed_dim_caps = [64]

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
    if args.candidate_offset < 0:
        raise ValueError("--candidate-offset must be >= 0.")
    if args.candidate_offset >= len(candidates):
        raise ValueError(f"--candidate-offset={args.candidate_offset} is >= candidates={len(candidates)}.")
    end = args.candidate_offset + max(1, args.max_runs)
    runs = candidates[args.candidate_offset : min(end, len(candidates))]

    # We write rows to CSV as we go (so if a long run is interrupted you still have results).
    fieldnames = [
        "source",
        "run",
        "config_id",
        "data",
        "feature_set",
        "seed",
        "split_seed",
        "split_strategy",
        "candidate_offset",
        "train_max_samples",
        "n_train_full",
        "n_train_used",
        "n_val",
        "n_test",
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
    ]

    # If resuming, skip the configs we already wrote to disk.
    start_run = 1
    if args.resume:
        try:
            with open(args.out_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing = reader.fieldnames or []
                if existing and existing != fieldnames:
                    raise ValueError(
                        "--resume: output CSV header does not match the current script. "
                        "Use a new --out-csv (or delete the old file) to start fresh."
                    )
                already = sum(1 for _ in reader)
            start_run = already + 1
        except FileNotFoundError:
            start_run = 1

    if start_run > len(runs):
        raise ValueError(f"--resume: output already has {start_run-1} rows, but only {len(runs)} runs requested.")

    extra = f", train_max_samples={args.train_max_samples}" if args.train_max_samples else ""
    print(
        f"Test-focused sweep: runs={len(runs)} / candidates={len(candidates)} "
        f"(epochs={args.epochs}, split_strategy={args.split_strategy}, seed={args.seed}, offset={args.candidate_offset}, split=70/15/15{extra})"
    )

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
                "config_id": _config_id(cfg),
                "data": args.data,
                "feature_set": payload.get("feature_set", ""),
                "seed": args.seed,
                "split_seed": args.seed,
                "split_strategy": args.split_strategy,
                "candidate_offset": int(args.candidate_offset),
                "train_max_samples": "" if args.train_max_samples is None else int(args.train_max_samples),
                "n_train_full": int(n_train_full),
                "n_train_used": int(n_train_used),
                "n_val": int(n_val),
                "n_test": int(n_test),
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
                "seconds": round(elapsed_s, 2),
            }

            w.writerow(row)
            f.flush()

            print(
                f"[{i:03d}/{len(runs)}] {cfg['name']:<6} hidden={row['hidden_dims']:<24} "
                f"bs={row['batch_size']:<4} lr={row['lr']:<6} do={row['dropout']:<4} wd={row['weight_decay']:<6} "
                f"test_rmse={float(row['test_rmse']):.4f}  test_acc={float(row['test_acc_pct']):.2f}%"
                f" (last={float(row['test_acc_pct_last']):.2f}%)  ({row['seconds']}s)"
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
            f"  test_rmse={float(r['test_rmse']):.4f}  test_acc={float(r['test_acc_pct']):.2f}% (last={float(r['test_acc_pct_last']):.2f}%)  val_rmse={float(r['val_rmse']):.4f}  "
            f"name={r['name']} hidden={r['hidden_dims']} bs={r['batch_size']} lr={r['lr']} do={r['dropout']} wd={r['weight_decay']} emb_cap={r['embed_dim_cap']}"
        )

    print("\nSaved results:", args.out_csv)


if __name__ == "__main__":
    main()
