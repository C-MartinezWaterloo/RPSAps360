#!/usr/bin/env python3
"""
Evaluation suite: rerun best configs across multiple random seeds and split seeds.

Goal:
  - Strengthen experimental evaluation (robustness across partitions).
  - Test time-based splits (future generalization).
  - Append results to a single intermediate log (gitignored), then merge into
    the canonical `results_all.csv` via `python export_results.py`.

This script intentionally avoids pandas and uses only existing project code.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from csv_utils import append_row
from hedonic_baseline import train_and_eval as hedonic_train_and_eval
from train_ann import make_splits as make_splits, train_and_eval as ann_train_and_eval


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


def _parse_str_list(s: str) -> list[str]:
    out = [p.strip() for p in s.split(",") if p.strip()]
    if not out:
        raise ValueError("Expected a comma-separated list.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ann_tensors_full.pt")
    parser.add_argument("--out-csv", default="eval_runs.csv")

    parser.add_argument(
        "--models",
        default="ann,hedonic",
        help="Comma-separated models to run: ann,hedonic (more later).",
    )
    parser.add_argument("--split-strategies", default="random,time", help="Comma-separated: random,time")
    parser.add_argument("--split-seeds", default="42,7,123", help="Comma-separated integers.")
    parser.add_argument("--model-seeds", default="0,1", help="Comma-separated integers (init + dataloader seed).")

    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)

    # Best-known ANN config (full features).
    parser.add_argument("--ann-hidden-dims", default="512,256,128,64")
    parser.add_argument("--ann-dropout", type=float, default=0.1)
    parser.add_argument("--ann-embed-dim-cap", type=int, default=64)
    parser.add_argument("--ann-epochs", type=int, default=30)
    parser.add_argument("--ann-batch-size", type=int, default=512)
    parser.add_argument("--ann-lr", type=float, default=0.004)
    parser.add_argument("--ann-weight-decay", type=float, default=0.001)

    # Hedonic baseline config (full features).
    parser.add_argument("--hedonic-epochs", type=int, default=20)
    parser.add_argument("--hedonic-batch-size", type=int, default=8192)
    parser.add_argument("--hedonic-lr", type=float, default=0.01)
    parser.add_argument("--hedonic-weight-decay", type=float, default=0.0)

    args = parser.parse_args()

    models = set(_parse_str_list(args.models))
    split_strategies = _parse_str_list(args.split_strategies)
    split_seeds = _parse_int_list(args.split_seeds)
    model_seeds = _parse_int_list(args.model_seeds)

    out_path = Path(args.out_csv)
    payload = torch.load(args.data, map_location="cpu")
    feature_set = payload.get("feature_set", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann_hidden_dims = _parse_int_list(args.ann_hidden_dims)

    run = 0
    started = time.time()

    for split_strategy in split_strategies:
        for split_seed in split_seeds:
            # Time split is deterministic; split_seed only affects within-bucket tie breaks.
            train_idx, val_idx, test_idx = make_splits(
                payload=payload,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=split_seed,
                strategy=split_strategy,
            )

            for model_seed in model_seeds:
                # Keep time runs smaller by default (since split_seed doesn't change much).
                if split_strategy == "time" and split_seed != split_seeds[0]:
                    continue

                if "ann" in models:
                    run += 1
                    t0 = time.time()
                    metrics = ann_train_and_eval(
                        payload=payload,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        test_idx=test_idx,
                        batch_size=args.ann_batch_size,
                        epochs=args.ann_epochs,
                        lr=args.ann_lr,
                        weight_decay=args.ann_weight_decay,
                        hidden_dims=ann_hidden_dims,
                        dropout=args.ann_dropout,
                        embed_dim_cap=args.ann_embed_dim_cap,
                        seed=model_seed,
                        device=device,
                        compute_test=True,
                    )
                    seconds = time.time() - t0
                    row = {
                        "source": "eval_suite",
                        "stage": "robustness",
                        "run": run,
                        "run_name": "ann_best_full",
                        "data": args.data,
                        "feature_set": feature_set,
                        "model": "ann_mlp",
                        "split_strategy": split_strategy,
                        "split_seed": split_seed,
                        "seed": model_seed,
                        "train_frac": args.train_frac,
                        "val_frac": args.val_frac,
                        "test_frac": args.test_frac,
                        "hidden_dims": ",".join(str(x) for x in ann_hidden_dims),
                        "dropout": args.ann_dropout,
                        "embed_dim_cap": args.ann_embed_dim_cap,
                        "epochs": args.ann_epochs,
                        "batch_size": args.ann_batch_size,
                        "lr": args.ann_lr,
                        "weight_decay": args.ann_weight_decay,
                        "seconds": round(seconds, 2),
                        **metrics,
                    }
                    append_row(out_path, row)
                    print(
                        f"[ann] run={run:>3} split={split_strategy:>6} split_seed={split_seed:<4} seed={model_seed:<2}"
                        f"  test_rmse={metrics['test_rmse']:.4f}  test_acc={metrics['test_acc_pct']:.2f}%  sec={seconds:.1f}"
                    )

                if "hedonic" in models:
                    run += 1
                    t0 = time.time()
                    metrics = hedonic_train_and_eval(
                        payload=payload,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        test_idx=test_idx,
                        epochs=args.hedonic_epochs,
                        batch_size=args.hedonic_batch_size,
                        lr=args.hedonic_lr,
                        weight_decay=args.hedonic_weight_decay,
                        seed=model_seed,
                        device=device,
                        compute_test=True,
                    )
                    seconds = time.time() - t0
                    row = {
                        "source": "eval_suite",
                        "stage": "robustness",
                        "run": run,
                        "run_name": "hedonic_full",
                        "data": args.data,
                        "feature_set": feature_set,
                        "model": "hedonic_linear_fixed_effects",
                        "split_strategy": split_strategy,
                        "split_seed": split_seed,
                        "seed": model_seed,
                        "train_frac": args.train_frac,
                        "val_frac": args.val_frac,
                        "test_frac": args.test_frac,
                        "epochs": args.hedonic_epochs,
                        "batch_size": args.hedonic_batch_size,
                        "lr": args.hedonic_lr,
                        "weight_decay": args.hedonic_weight_decay,
                        "seconds": round(seconds, 2),
                        **metrics,
                    }
                    append_row(out_path, row)
                    print(
                        f"[hed] run={run:>3} split={split_strategy:>6} split_seed={split_seed:<4} seed={model_seed:<2}"
                        f"  test_rmse={metrics['test_rmse']:.4f}  test_acc={metrics['test_acc_pct']:.2f}%  sec={seconds:.1f}"
                    )

    elapsed = time.time() - started
    print(f"\nDone. Wrote: {out_path}  elapsed={elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

