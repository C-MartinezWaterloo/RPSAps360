#!/usr/bin/env python3
"""
Run a larger hyperparameter sweep for the ANN.

Important: this sweep is TEST-CLEAN.
  - We train and rank configs using *validation* only.
  - The test set is evaluated exactly once, at the very end, for the single
    best config found by validation.

What it does:
  - Loads `ann_tensors.pt` once.
  - Creates a single deterministic train/val/test split.
  - Phase 1: try many configs for a small number of epochs (validation only).
  - Phase 2: re-train the top configs longer (validation only).
  - Final: train once on (train + val) and evaluate on test (one time).
  - Writes a CSV file you can open in Excel.

Metric:
  - We report MSE/RMSE on log1p(TransactionPrice). This is effectively RMSLE.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from typing import Any

import torch

from train_ann import _make_splits, train_and_eval, train_fixed_epochs_eval_test


def _fmt_dims(hidden_dims: list[int]) -> str:
    return ",".join(str(x) for x in hidden_dims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors.pt", help="Output from prepare_ann_tensors.py")
    parser.add_argument("--out-csv", default="sweep_results_clean.csv")
    parser.add_argument("--sweep-epochs", type=int, default=3, help="Epochs for phase 1 (many configs)")
    parser.add_argument("--refine-epochs", type=int, default=10, help="Epochs for phase 2 (top configs)")
    parser.add_argument("--max-runs", type=int, default=120, help="How many configs to try in phase 1")
    parser.add_argument("--refine-top", type=int, default=12, help="How many configs to re-train in phase 2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10, help="How many configs to print at the end (by val)")
    args = parser.parse_args()

    payload = torch.load(args.data, map_location="cpu")
    n = payload["X_num"].shape[0]

    # Fixed split for fair comparisons.
    train_idx, val_idx, test_idx = _make_splits(n, 0.70, 0.15, 0.15, args.seed)

    # --- Config space --------------------------------------------------------
    # We build a large *candidate* grid, then shuffle+take the first N runs.
    # This gives "lots of variety" without forcing thousands of runs.
    hidden_spaces = [
        ("small2", [128, 64]),
        ("base2", [256, 128]),
        ("wide2", [512, 256]),
        ("deep3", [512, 256, 128]),
        ("deep4", [512, 256, 128, 64]),
        ("big3", [1024, 512, 256]),
        ("big4", [1024, 512, 256, 128]),
    ]
    batch_sizes = [1024, 2048, 4096, 8192, 16384]
    lrs = [0.0005, 0.001, 0.002, 0.003, 0.004]
    dropouts = [0.0, 0.05, 0.1, 0.2]
    weight_decays = [0.0, 1e-4, 1e-3]
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

    phase1 = candidates[: max(1, min(args.max_runs, len(candidates)))]
    print(
        f"Phase 1: {len(phase1)} configs (epochs={args.sweep_epochs}). "
        f"Candidate grid size={len(candidates)}. Split seed={args.seed}."
    )

    results: list[dict[str, Any]] = []

    # ------------------------------ PHASE 1 ---------------------------------
    for i, cfg in enumerate(phase1, start=1):
        start = time.time()

        metrics = train_and_eval(
            payload=payload,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=int(cfg["batch_size"]),
            epochs=args.sweep_epochs,
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            hidden_dims=list(cfg["hidden_dims"]),
            dropout=float(cfg["dropout"]),
            embed_dim_cap=int(cfg["embed_dim_cap"]),
            seed=args.seed,
            compute_test=False,  # <-- test-clean
        )

        elapsed_s = time.time() - start

        row = {
            "stage": "phase1",
            "run": i,
            "seed": args.seed,
            "split_seed": args.seed,
            "name": cfg["name"],
            "hidden_dims": _fmt_dims(cfg["hidden_dims"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["lr"]),
            "dropout": float(cfg["dropout"]),
            "weight_decay": float(cfg["weight_decay"]),
            "embed_dim_cap": int(cfg["embed_dim_cap"]),
            "epochs": int(args.sweep_epochs),
            "best_val_epoch": metrics["best_val_epoch"],
            "best_val_mse": metrics["best_val_mse"],
            "train_mse": metrics["train_mse"],
            "val_mse": metrics["val_mse"],
            "test_mse": "",  # test is not computed in phase 1
            "train_rmse": metrics["train_rmse"],
            "val_rmse": metrics["val_rmse"],
            "test_rmse": "",
            "n_params": metrics["n_params"],
            "seconds": round(elapsed_s, 2),
        }
        results.append(row)

        print(
            f"[P1 {i:03d}/{len(phase1)}] {cfg['name']:<6} hidden={row['hidden_dims']:<18} "
            f"bs={row['batch_size']:<5} lr={row['lr']:<6} do={row['dropout']:<4} wd={row['weight_decay']:<6} "
            f"val_rmse={float(row['val_rmse']):.4f} ({row['seconds']}s)"
        )

    # Pick top configs by validation RMSE (lower is better).
    phase1_sorted = sorted(
        [r for r in results if r["stage"] == "phase1"],
        key=lambda r: float(r["val_rmse"]),
    )
    refine_top = phase1_sorted[: max(1, min(args.refine_top, len(phase1_sorted)))]

    # ------------------------------ PHASE 2 ---------------------------------
    print(f"\nPhase 2: re-train top {len(refine_top)} configs longer (epochs={args.refine_epochs}).")
    for j, r in enumerate(refine_top, start=1):
        cfg = {
            "name": r["name"],
            "hidden_dims": [int(x) for x in str(r["hidden_dims"]).split(",") if x],
            "batch_size": int(r["batch_size"]),
            "lr": float(r["lr"]),
            "dropout": float(r["dropout"]),
            "weight_decay": float(r["weight_decay"]),
            "embed_dim_cap": int(r["embed_dim_cap"]),
        }

        start = time.time()
        metrics = train_and_eval(
            payload=payload,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=cfg["batch_size"],
            epochs=args.refine_epochs,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
            embed_dim_cap=cfg["embed_dim_cap"],
            seed=args.seed,
            compute_test=False,  # still test-clean
        )
        elapsed_s = time.time() - start

        row = {
            "stage": "phase2",
            "run": j,
            "seed": args.seed,
            "split_seed": args.seed,
            "name": cfg["name"],
            "hidden_dims": _fmt_dims(cfg["hidden_dims"]),
            "batch_size": cfg["batch_size"],
            "lr": cfg["lr"],
            "dropout": cfg["dropout"],
            "weight_decay": cfg["weight_decay"],
            "embed_dim_cap": cfg["embed_dim_cap"],
            "epochs": int(args.refine_epochs),
            "best_val_epoch": metrics["best_val_epoch"],
            "best_val_mse": metrics["best_val_mse"],
            "train_mse": metrics["train_mse"],
            "val_mse": metrics["val_mse"],
            "test_mse": "",
            "train_rmse": metrics["train_rmse"],
            "val_rmse": metrics["val_rmse"],
            "test_rmse": "",
            "n_params": metrics["n_params"],
            "seconds": round(elapsed_s, 2),
        }
        results.append(row)

        print(
            f"[P2 {j:02d}/{len(refine_top)}] {cfg['name']:<6} hidden={row['hidden_dims']:<18} "
            f"bs={cfg['batch_size']:<5} lr={cfg['lr']:<6} do={cfg['dropout']:<4} wd={cfg['weight_decay']:<6} "
            f"val_rmse={float(row['val_rmse']):.4f} ({row['seconds']}s)"
        )

    phase2_rows = [r for r in results if r["stage"] == "phase2"]
    best_phase2 = min(phase2_rows, key=lambda r: float(r["val_rmse"]))

    best_cfg = {
        "name": best_phase2["name"],
        "hidden_dims": [int(x) for x in str(best_phase2["hidden_dims"]).split(",") if x],
        "batch_size": int(best_phase2["batch_size"]),
        "lr": float(best_phase2["lr"]),
        "dropout": float(best_phase2["dropout"]),
        "weight_decay": float(best_phase2["weight_decay"]),
        "embed_dim_cap": int(best_phase2["embed_dim_cap"]),
        "epochs": int(best_phase2["best_val_epoch"]),
    }

    # ------------------------------ FINAL -----------------------------------
    # Final, clean test evaluation:
    #   - Train on (train + val)
    #   - Train for a fixed number of epochs chosen from validation (best_val_epoch)
    #   - Evaluate on test ONCE.
    print("\nFinal (test-clean): train on train+val, evaluate on test once.")
    trainval_idx = train_idx + val_idx
    start = time.time()
    final_metrics = train_fixed_epochs_eval_test(
        payload=payload,
        train_idx=trainval_idx,
        test_idx=test_idx,
        batch_size=best_cfg["batch_size"],
        epochs=best_cfg["epochs"],
        lr=best_cfg["lr"],
        weight_decay=best_cfg["weight_decay"],
        hidden_dims=best_cfg["hidden_dims"],
        dropout=best_cfg["dropout"],
        embed_dim_cap=best_cfg["embed_dim_cap"],
        seed=args.seed,
    )
    elapsed_s = time.time() - start

    results.append(
        {
            "stage": "final_test",
            "run": 1,
            "seed": args.seed,
            "split_seed": args.seed,
            "name": best_cfg["name"],
            "hidden_dims": _fmt_dims(best_cfg["hidden_dims"]),
            "batch_size": best_cfg["batch_size"],
            "lr": best_cfg["lr"],
            "dropout": best_cfg["dropout"],
            "weight_decay": best_cfg["weight_decay"],
            "embed_dim_cap": best_cfg["embed_dim_cap"],
            "epochs": best_cfg["epochs"],
            "best_val_epoch": best_cfg["epochs"],
            "best_val_mse": "",
            "train_mse": final_metrics["train_mse"],
            "val_mse": "",
            "test_mse": final_metrics["test_mse"],
            "train_rmse": final_metrics["train_rmse"],
            "val_rmse": "",
            "test_rmse": final_metrics["test_rmse"],
            "n_params": final_metrics["n_params"],
            "seconds": round(elapsed_s, 2),
        }
    )

    # ------------------------------ OUTPUT ----------------------------------
    # Write CSV so you can sort/filter in Excel easily.
    fieldnames = list(results[0].keys()) if results else []
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print("\nTop configs by validation RMSE (phase 1):")
    for r in phase1_sorted[: args.top_k]:
        print(
            f"  val_rmse={float(r['val_rmse']):.4f}  name={r['name']} hidden={r['hidden_dims']} "
            f"bs={r['batch_size']} lr={r['lr']} do={r['dropout']} wd={r['weight_decay']} emb_cap={r['embed_dim_cap']}"
        )

    print("\nBest config by validation (phase 2):")
    print(
        f"  val_rmse={float(best_phase2['val_rmse']):.4f}  name={best_cfg['name']} hidden={_fmt_dims(best_cfg['hidden_dims'])} "
        f"bs={best_cfg['batch_size']} lr={best_cfg['lr']} do={best_cfg['dropout']} wd={best_cfg['weight_decay']} emb_cap={best_cfg['embed_dim_cap']} "
        f"best_val_epoch={best_cfg['epochs']}"
    )

    print("\nFinal test result (evaluated once):")
    print(f"  test_rmse={final_metrics['test_rmse']:.4f}  test_mse={final_metrics['test_mse']:.6f}")
    print("\nSaved results:", args.out_csv)


if __name__ == "__main__":
    main()
