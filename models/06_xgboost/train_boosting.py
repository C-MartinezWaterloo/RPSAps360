#!/usr/bin/env python3
"""
Train gradient-boosted tree baselines (XGBoost / LightGBM) on this project's
tensor format.

Why this exists:
  - XGBoost / LightGBM are strong tabular baselines.
  - This repo's default environment intentionally avoids compiled deps, so these
    libraries may not be installed here.
  - This script is designed to be run in a separate Python environment where
    `xgboost` and/or `lightgbm` are available, and then merged into the single
    canonical sheet via `python export_results.py`.

Data + target:
  - Inputs come from `prepare_ann_tensors.py` outputs (e.g. `ann_tensors_full.pt`)
  - Target is `y_log1p = log1p(TransactionPrice)`

Categoricals:
  - We avoid one-hotting high-cardinality categoricals (too wide without SciPy).
  - Instead, we build compact numeric features from categoricals using:
      * frequency encoding (log1p(count))
      * target mean encoding (optionally K-fold within training)

Metrics (consistent with the rest of the repo):
  - MSE/RMSE in log1p(price) space
  - MdAPE + Accuracy% in price space, where accuracy% = max(0, 100*(1 - MdAPE))
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from csv_utils import append_row
from train_ann import make_splits


def _metrics_from_preds(y_log1p: np.ndarray, pred_log1p: np.ndarray) -> dict[str, float]:
    y_log1p = y_log1p.reshape(-1)
    pred_log1p = pred_log1p.reshape(-1)

    mse = float(np.mean((pred_log1p - y_log1p) ** 2))
    rmse = float(mse**0.5)

    eps = 1e-8
    y_true = np.expm1(y_log1p)
    y_pred = np.expm1(pred_log1p)
    ape = np.abs(y_pred - y_true) / np.clip(y_true, eps, None)
    mdape = float(np.median(ape))
    acc_pct = float(max(0.0, 100.0 * (1.0 - mdape)))

    return {"mse": mse, "rmse": rmse, "mdape": mdape, "acc_pct": acc_pct}


def _target_encode_full(
    cat_ids: np.ndarray,
    y: np.ndarray,
    *,
    cat_size: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a smoothed target-mean encoding map from a training set.

    Returns:
      (enc_map, counts, sums)
    """
    cat_ids = cat_ids.astype(np.int64, copy=False).reshape(-1)
    y = y.astype(np.float64, copy=False).reshape(-1)

    counts = np.bincount(cat_ids, minlength=cat_size).astype(np.float64)
    sums = np.bincount(cat_ids, weights=y, minlength=cat_size).astype(np.float64)
    global_mean = float(np.mean(y)) if y.size else 0.0

    enc_map = (sums + smoothing * global_mean) / (counts + smoothing)
    return enc_map.astype(np.float32), counts, sums


def _target_encode_kfold_train(
    cat_ids_train: np.ndarray,
    y_train: np.ndarray,
    *,
    cat_size: int,
    smoothing: float,
    k_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    K-fold target encoding for the training rows only (reduces target leakage).

    Also returns full-training (counts, sums) for later building val/test maps.
    """
    cat_ids_train = cat_ids_train.astype(np.int64, copy=False).reshape(-1)
    y_train = y_train.astype(np.float64, copy=False).reshape(-1)

    n = int(cat_ids_train.shape[0])
    if k_folds <= 1 or n <= 1:
        enc_map, counts, sums = _target_encode_full(cat_ids_train, y_train, cat_size=cat_size, smoothing=smoothing)
        return enc_map[cat_ids_train], counts, sums

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k_folds)

    total_counts = np.bincount(cat_ids_train, minlength=cat_size).astype(np.float64)
    total_sums = np.bincount(cat_ids_train, weights=y_train, minlength=cat_size).astype(np.float64)
    global_mean = float(np.mean(y_train)) if y_train.size else 0.0

    enc_out = np.empty(n, dtype=np.float32)
    for fold_idx in folds:
        fold_c = np.bincount(cat_ids_train[fold_idx], minlength=cat_size).astype(np.float64)
        fold_s = np.bincount(cat_ids_train[fold_idx], weights=y_train[fold_idx], minlength=cat_size).astype(np.float64)

        out_counts = total_counts - fold_c
        out_sums = total_sums - fold_s
        enc_map = (out_sums + smoothing * global_mean) / (out_counts + smoothing)
        enc_out[fold_idx] = enc_map[cat_ids_train[fold_idx]].astype(np.float32)

    return enc_out, total_counts, total_sums


def _build_features(
    *,
    payload: dict,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    cat_encoding: str,
    k_folds: int,
    smoothing: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x_num = payload["X_num"].detach().cpu().numpy().astype(np.float32)
    x_cat = payload["X_cat"].detach().cpu().numpy().astype(np.int64)
    y_log1p = payload["y_log1p"].detach().cpu().numpy().astype(np.float32).reshape(-1)

    x_num_train = x_num[train_idx]
    x_num_val = x_num[val_idx]
    x_num_test = x_num[test_idx]
    y_train = y_log1p[train_idx]
    y_val = y_log1p[val_idx]
    y_test = y_log1p[test_idx]

    feature_names: list[str] = [f"num:{c}" for c in payload.get("numeric_cols", [])]
    feats_train = [x_num_train]
    feats_val = [x_num_val]
    feats_test = [x_num_test]

    if cat_encoding == "none":
        return (
            np.concatenate(feats_train, axis=1),
            np.concatenate(feats_val, axis=1),
            np.concatenate(feats_test, axis=1),
            feature_names,
        )

    cat_cols: list[str] = list(payload.get("cat_cols", []))
    cat_maps: dict = payload.get("cat_maps", {})
    cat_sizes = [len(cat_maps[c]) for c in cat_cols]

    x_cat_train = x_cat[train_idx]
    x_cat_val = x_cat[val_idx]
    x_cat_test = x_cat[test_idx]

    for j, col in enumerate(cat_cols):
        cat_size = int(cat_sizes[j])
        ids_tr = x_cat_train[:, j]
        ids_va = x_cat_val[:, j]
        ids_te = x_cat_test[:, j]

        if "target" in cat_encoding:
            enc_tr, counts, sums = _target_encode_kfold_train(
                ids_tr,
                y_train,
                cat_size=cat_size,
                smoothing=smoothing,
                k_folds=(k_folds if "kfold" in cat_encoding else 1),
                seed=seed + 17 * (j + 1),
            )
            global_mean = float(np.mean(y_train)) if y_train.size else 0.0
            enc_map = (sums + smoothing * global_mean) / (counts + smoothing)
            enc_va = enc_map[ids_va].astype(np.float32)
            enc_te = enc_map[ids_te].astype(np.float32)

            feats_train.append(enc_tr.reshape(-1, 1))
            feats_val.append(enc_va.reshape(-1, 1))
            feats_test.append(enc_te.reshape(-1, 1))
            feature_names.append(f"cat_target:{col}")

        if "freq" in cat_encoding:
            counts = np.bincount(ids_tr.astype(np.int64, copy=False), minlength=cat_size).astype(np.float64)
            freq_map = np.log1p(counts).astype(np.float32)
            feats_train.append(freq_map[ids_tr].reshape(-1, 1))
            feats_val.append(freq_map[ids_va].reshape(-1, 1))
            feats_test.append(freq_map[ids_te].reshape(-1, 1))
            feature_names.append(f"cat_freq:{col}")

    return (
        np.concatenate(feats_train, axis=1),
        np.concatenate(feats_val, axis=1),
        np.concatenate(feats_test, axis=1),
        feature_names,
    )


def _require_lib(model: str):
    if model == "xgboost":
        try:
            import xgboost as xgb  # type: ignore

            return xgb
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "xgboost is not available in this environment.\n"
                "Install it in a separate env (example):\n"
                "  pip install xgboost\n"
                f"Import error: {type(e).__name__}: {e}"
            )
    if model == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore

            return lgb
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "lightgbm is not available in this environment.\n"
                "Install it in a separate env (example):\n"
                "  pip install lightgbm\n"
                f"Import error: {type(e).__name__}: {e}"
            )
    raise ValueError(f"Unknown model: {model}")


def _train_xgboost(
    *,
    xgb,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
    num_rounds: int,
) -> tuple[dict, dict]:
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dtest = xgb.DMatrix(x_test, label=y_test)

    evals_result: dict = {}
    bst_full = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    val_rmse_hist = evals_result.get("val", {}).get("rmse", [])
    if not val_rmse_hist:
        best_iter = num_rounds - 1
        best_val_rmse = float("nan")
    else:
        best_iter = int(np.argmin(np.array(val_rmse_hist, dtype=np.float64)))
        best_val_rmse = float(val_rmse_hist[best_iter])

    # Train a second booster restricted to the best iteration so predictions match "best checkpoint"
    # without relying on version-specific predict kwargs.
    bst_best = xgb.train(
        params,
        dtrain,
        num_boost_round=int(best_iter + 1),
        evals=[(dval, "val")],
        verbose_eval=False,
    )

    pred = {}
    pred["train_best"] = bst_best.predict(dtrain)
    pred["val_best"] = bst_best.predict(dval)
    pred["test_best"] = bst_best.predict(dtest)
    pred["train_last"] = bst_full.predict(dtrain)
    pred["val_last"] = bst_full.predict(dval)
    pred["test_last"] = bst_full.predict(dtest)

    best = {
        "best_iteration": int(best_iter + 1),
        "best_val_rmse": float(best_val_rmse),
        "pred": pred,
    }
    last = {
        "last_iteration": int(num_rounds),
    }
    return best, last


def _train_lightgbm(
    *,
    lgb,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
    num_rounds: int,
    early_stopping_rounds: int | None,
) -> tuple[dict, dict]:
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)

    callbacks = []
    if early_stopping_rounds and early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False))

    bst_full = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=int(num_rounds),
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    best_iter = int(getattr(bst_full, "best_iteration", 0) or 0)
    if best_iter <= 0:
        best_iter = int(bst_full.current_iteration())

    pred = {}
    pred["train_best"] = bst_full.predict(x_train, num_iteration=best_iter)
    pred["val_best"] = bst_full.predict(x_val, num_iteration=best_iter)
    pred["test_best"] = bst_full.predict(x_test, num_iteration=best_iter)
    pred["train_last"] = bst_full.predict(x_train)
    pred["val_last"] = bst_full.predict(x_val)
    pred["test_last"] = bst_full.predict(x_test)

    best = {"best_iteration": int(best_iter), "pred": pred}
    last = {"last_iteration": int(bst_full.current_iteration())}
    return best, last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "lightgbm"], required=True)
    parser.add_argument("--data", default="ann_tensors_full.pt")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "time"],
        default="random",
        help="How to split data into train/val/test (default: random). 'time' sorts by TransactionYear/Quarter.",
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=0,
        help="Optional: cap training rows for speed. Use 0 for full training split.",
    )

    parser.add_argument(
        "--cat-encoding",
        choices=["none", "freq", "target", "target_kfold", "target_kfold+freq"],
        default="target_kfold+freq",
        help="How to turn categorical indices into numeric features (default: target_kfold+freq).",
    )
    parser.add_argument("--k-folds", type=int, default=5, help="K for K-fold target encoding (train-only).")
    parser.add_argument("--smoothing", type=float, default=20.0, help="Smoothing strength for target mean encoding.")

    parser.add_argument("--num-rounds", type=int, default=2000, help="Boosting rounds / estimators.")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="LightGBM early stopping rounds (0 to disable).")

    # Shared-ish knobs (mapped into each library where sensible).
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)

    # XGBoost-specific
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)

    # LightGBM-specific
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    args = parser.parse_args()

    if args.num_rounds <= 0:
        raise SystemExit("--num-rounds must be positive.")
    if args.train_max_samples < 0:
        raise SystemExit("--train-max-samples must be >= 0.")

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = "xgb_runs.csv" if args.model == "xgboost" else "lgbm_runs.csv"

    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.model}_{args.split_strategy}_seed{args.seed}"

    lib = _require_lib(args.model)

    payload = torch.load(args.data, map_location="cpu")
    feature_set = payload.get("feature_set", "")

    train_idx_full, val_idx, test_idx = make_splits(
        payload=payload,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        strategy=args.split_strategy,
    )

    train_idx = list(train_idx_full)
    n_train_full = len(train_idx_full)
    if args.train_max_samples and args.train_max_samples < len(train_idx_full):
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(np.array(train_idx_full, dtype=np.int64), size=int(args.train_max_samples), replace=False)
        train_idx = keep.tolist()

    x_train, x_val, x_test, feat_names = _build_features(
        payload=payload,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cat_encoding=args.cat_encoding,
        k_folds=args.k_folds,
        smoothing=args.smoothing,
        seed=args.seed,
    )

    y = payload["y_log1p"].detach().cpu().numpy().astype(np.float32).reshape(-1)
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    t0 = time.time()

    if args.model == "xgboost":
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": float(args.lr),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample),
            "min_child_weight": float(args.min_child_weight),
            "gamma": float(args.gamma),
            "lambda": float(args.reg_lambda),
            "alpha": float(args.reg_alpha),
            "seed": int(args.seed),
            "tree_method": "hist",
        }
        best, last = _train_xgboost(
            xgb=lib,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            params=params,
            num_rounds=int(args.num_rounds),
        )
    else:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": float(args.lr),
            "max_depth": int(args.max_depth),
            "num_leaves": int(args.num_leaves),
            "min_data_in_leaf": int(args.min_data_in_leaf),
            "feature_fraction": float(args.feature_fraction),
            "bagging_fraction": float(args.bagging_fraction),
            "bagging_freq": int(args.bagging_freq),
            "seed": int(args.seed),
            "verbosity": -1,
        }
        best, last = _train_lightgbm(
            lgb=lib,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            params=params,
            num_rounds=int(args.num_rounds),
            early_stopping_rounds=(None if args.early_stopping_rounds <= 0 else int(args.early_stopping_rounds)),
        )

    seconds = float(time.time() - t0)

    pred = best["pred"]
    train_best = _metrics_from_preds(y_train, pred["train_best"])
    val_best = _metrics_from_preds(y_val, pred["val_best"])
    test_best = _metrics_from_preds(y_test, pred["test_best"])
    train_last = _metrics_from_preds(y_train, pred["train_last"])
    val_last = _metrics_from_preds(y_val, pred["val_last"])
    test_last = _metrics_from_preds(y_test, pred["test_last"])

    row = {
        "source": args.model,
        "stage": args.model,
        "run_name": run_name,
        "data": args.data,
        "feature_set": feature_set,
        "seed": args.seed,
        "split_seed": args.seed,
        "split_strategy": args.split_strategy,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "train_max_samples": ("" if not args.train_max_samples else int(args.train_max_samples)),
        "n_train_full": int(n_train_full),
        "n_train_used": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "epochs": int(args.num_rounds),
        "lr": float(args.lr),
        "max_depth": int(args.max_depth),
        "subsample": float(args.subsample),
        "colsample": float(args.colsample),
        "cat_encoding": args.cat_encoding,
        "k_folds": int(args.k_folds),
        "smoothing": float(args.smoothing),
        "best_val_epoch": int(best.get("best_iteration", 0) or 0),
        "best_val_mse": float(val_best["mse"]),
        "train_mse": float(train_best["mse"]),
        "val_mse": float(val_best["mse"]),
        "test_mse": float(test_best["mse"]),
        "train_rmse": float(train_best["rmse"]),
        "val_rmse": float(val_best["rmse"]),
        "test_rmse": float(test_best["rmse"]),
        "train_mdape": float(train_best["mdape"]),
        "val_mdape": float(val_best["mdape"]),
        "test_mdape": float(test_best["mdape"]),
        "train_acc_pct": float(train_best["acc_pct"]),
        "val_acc_pct": float(val_best["acc_pct"]),
        "test_acc_pct": float(test_best["acc_pct"]),
        "train_mse_last": float(train_last["mse"]),
        "val_mse_last": float(val_last["mse"]),
        "test_mse_last": float(test_last["mse"]),
        "train_rmse_last": float(train_last["rmse"]),
        "val_rmse_last": float(val_last["rmse"]),
        "test_rmse_last": float(test_last["rmse"]),
        "train_mdape_last": float(train_last["mdape"]),
        "val_mdape_last": float(val_last["mdape"]),
        "test_mdape_last": float(test_last["mdape"]),
        "train_acc_pct_last": float(train_last["acc_pct"]),
        "val_acc_pct_last": float(val_last["acc_pct"]),
        "test_acc_pct_last": float(test_last["acc_pct"]),
        "seconds": round(seconds, 2),
        "model": args.model,
        "notes": "Boosting baseline with compact categorical encodings (no one-hot).",
    }

    out_path = Path(out_csv)
    append_row(out_path, row)
    print(f"Wrote CSV row: {out_path}")


if __name__ == "__main__":
    main()
