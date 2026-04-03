#!/usr/bin/env python3
"""
Export all experiment results into a single CSV.

Goal (per your request):
  - Keep ONE canonical results file for GitHub: `results_all.csv`
  - Keep *all* results ever recorded in that file, even if intermediate logs
    are deleted later.
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

def _infer_feature_set(row: dict) -> str:
    data = str(row.get("data", "")).strip()
    feature_set = str(row.get("feature_set", "")).strip()
    if feature_set:
        return feature_set
    # Backfill for older tensors that didn't store feature_set (ex: ann_tensors.pt).
    if "basic_time" in data:
        return "basic_time"
    if "full" in data:
        return "full"
    if data.endswith("ann_tensors.pt"):
        return "basic"
    return ""


def _ensure_defaults(row: dict) -> None:
    """
    Ensure rows have consistent minimal metadata for filtering/plotting.

    We intentionally do *not* overwrite values that already exist in the row.
    """
    source = str(row.get("source", "")).strip()

    # Stage defaults (keeps sheets readable across many result sources).
    if "stage" not in row or not str(row.get("stage", "")).strip():
        if source in {"manual_runs"}:
            row["stage"] = "manual"
        elif source in {"train_ann"}:
            row["stage"] = "single_run"
        elif source in {"sweep_dirty"}:
            row["stage"] = "sweep_dirty"
        elif source in {"sweep_clean"}:
            row["stage"] = "sweep_clean"
        elif source in {"sweep_deep_test"}:
            row["stage"] = "sweep_deep_test"
        elif source in {"hedonic"}:
            row["stage"] = "hedonic_baseline"
        elif source in {"train_fm", "sweep_fm"}:
            row["stage"] = "fm"
        elif source in {"train_deepfm", "sweep_deepfm"}:
            row["stage"] = "deepfm"
        elif source in {"train_tabnet", "sweep_tabnet"}:
            row["stage"] = "tabnet"
        elif source in {"train_fttransformer", "sweep_fttransformer"}:
            row["stage"] = "fttransformer"
        elif source in {"eval_suite"}:
            row["stage"] = "robustness"
        elif source in {"xgboost"}:
            row["stage"] = "xgboost"
        elif source in {"lightgbm"}:
            row["stage"] = "lightgbm"

    # Model defaults (so you can quickly group by family).
    if "model" not in row or not str(row.get("model", "")).strip():
        if source in {"manual_runs", "train_ann", "sweep_dirty", "sweep_clean", "sweep_deep_test"}:
            row["model"] = "ann_mlp"
        elif source == "hedonic":
            row["model"] = "hedonic_linear_fixed_effects"
        elif source in {"train_fm", "sweep_fm"}:
            row["model"] = "factorization_machine"
        elif source in {"train_deepfm", "sweep_deepfm"}:
            row["model"] = "deepfm"
        elif source in {"train_tabnet", "sweep_tabnet"}:
            row["model"] = "tabnet"
        elif source in {"train_fttransformer", "sweep_fttransformer"}:
            row["model"] = "fttransformer"
        elif source in {"eval_suite"}:
            # Model should already be present, but keep a safe default.
            row["model"] = str(row.get("model", "")).strip() or "unknown"
        elif source in {"xgboost"}:
            row["model"] = "xgboost"
        elif source in {"lightgbm"}:
            row["model"] = "lightgbm"

    # Backfill feature_set when missing.
    inferred = _infer_feature_set(row)
    if inferred and (("feature_set" not in row) or (not str(row.get("feature_set", "")).strip())):
        row["feature_set"] = inferred

    # Backfill split metadata for older logs that omitted it.
    if "split_strategy" not in row or not str(row.get("split_strategy", "")).strip():
        row["split_strategy"] = "random"
    if "split_seed" not in row or not str(row.get("split_seed", "")).strip():
        seed = str(row.get("seed", "")).strip()
        if seed:
            row["split_seed"] = seed


def _row_signature(row: dict) -> tuple[tuple[str, str], ...]:
    """
    Stable, order-independent signature for de-duplicating rows across inputs.

    We ignore a small set of bookkeeping fields so the same run coming from
    different source files collapses to one row.
    """
    # `run_group` and `run` are bookkeeping fields that can change depending on which
    # intermediate CSV produced the row. `seconds` is also noisy across re-runs.
    # We ignore them so repeated ingests collapse to one row when all meaningful
    # fields match.
    ignore_keys = {"run_group", "run", "seconds"}
    items: list[tuple[str, str]] = []
    for k, v in row.items():
        if k in ignore_keys:
            continue
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        items.append((str(k).strip(), s))
    return tuple(sorted(items))


def main() -> None:
    root = Path(__file__).resolve().parent
    out_csv = root / "results_all.csv"

    rows_all: list[dict] = []

    # 0) Start from the existing canonical results file so we never lose history,
    # even if intermediate logs are deleted.
    for row in _load_csv_rows(out_csv):
        _ensure_defaults(row)
        rows_all.append(row)

    # These are the specific configs/metrics we ran earlier (before the big sweep)
    # using `train_ann.py` with a 70/15/15 split and seed=42.
    manual_runs: list[dict] = [
        {
            "run": 1,
            "notes": "3 epochs baseline",
            "data": "ann_tensors.pt",
            "feature_set": "basic",
            "split_seed": 42,
            "split_strategy": "random",
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
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
            "data": "ann_tensors.pt",
            "feature_set": "basic",
            "split_seed": 42,
            "split_strategy": "random",
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
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
            "data": "ann_tensors.pt",
            "feature_set": "basic",
            "split_seed": 42,
            "split_strategy": "random",
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
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
            "data": "ann_tensors.pt",
            "feature_set": "basic",
            "split_seed": 42,
            "split_strategy": "random",
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
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
            "data": "ann_tensors.pt",
            "feature_set": "basic",
            "split_seed": 42,
            "split_strategy": "random",
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
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
            "data": "ann_tensors.pt",
            "feature_set": "basic",
            "split_seed": 42,
            "split_strategy": "random",
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
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
            "data": "ann_tensors.pt",
            "feature_set": "basic",
            "split_seed": 42,
            "split_strategy": "random",
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
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
            "data": "ann_tensors.pt",
            "feature_set": "basic",
            "split_seed": 42,
            "split_strategy": "random",
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
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

    # 1) Manual runs (already numeric).
    for r in manual_runs:
        row = {"source": "manual_runs", "run_group": "manual_runs", **r}
        _ensure_defaults(row)
        rows_all.append(row)

    # 2) Logged single ANN runs.
    for path in sorted(root.glob("ann_runs*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "train_ann")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    # 3) Sweeps (all varieties, including historical tracked sweep_*.csv files).
    sweep_paths: set[Path] = set()
    sweep_paths.add(root / "sweep_results.csv")
    sweep_paths.update(root.glob("sweep_results_clean*.csv"))
    sweep_paths.update(root.glob("sweep_results_deep_test*.csv"))
    sweep_paths.update(root.glob("sweep_*.csv"))

    for path in sorted(p for p in sweep_paths if p.exists()):
        for row in _load_csv_rows(path):
            # Preserve explicit source if present; otherwise infer from filename.
            if "source" not in row or not str(row.get("source", "")).strip():
                if path.name == "sweep_results.csv":
                    row["source"] = "sweep_dirty"
                elif path.name.startswith("sweep_results_clean"):
                    row["source"] = "sweep_clean"
                else:
                    row["source"] = "sweep_deep_test"
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    # 6) Hedonic baseline.
    for path in sorted(root.glob("hedonic_results*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "hedonic")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    # 7) Next model(s): Factorization Machine.
    for path in sorted(root.glob("fm_runs*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "train_fm")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    for path in sorted(root.glob("fm_sweep*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "sweep_fm")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    # 8) DeepFM.
    for path in sorted(root.glob("deepfm_runs*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "train_deepfm")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    for path in sorted(root.glob("deepfm_sweep*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "sweep_deepfm")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    # 9) TabNet.
    for path in sorted(root.glob("tabnet_runs*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "train_tabnet")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    for path in sorted(root.glob("tabnet_sweep*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "sweep_tabnet")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    # 9.5) FT-Transformer (Transformer baseline for tabular).
    for path in sorted(root.glob("fttransformer_runs*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "train_fttransformer")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    for path in sorted(root.glob("fttransformer_sweep*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "sweep_fttransformer")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    # 10) Robustness eval suite (multi-seed, multi-split).
    for path in sorted(root.glob("eval_runs*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "eval_suite")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    # 11) External boosting baselines (run in a separate environment).
    for path in sorted(root.glob("xgb_runs*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "xgboost")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    for path in sorted(root.glob("lgbm_runs*.csv")):
        for row in _load_csv_rows(path):
            row.setdefault("source", "lightgbm")
            row.setdefault("run_group", path.stem)
            _ensure_defaults(row)
            rows_all.append(row)

    # De-duplicate (important now that we re-ingest results_all.csv as an input).
    deduped: list[dict] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for row in rows_all:
        sig = _row_signature(row)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(row)
    rows_all = deduped

    # Merge "schema-upgrade duplicates" for single-run style logs. These happen when
    # we add new metric columns over time and re-run the same config.
    def _is_blank(v: object) -> bool:
        return v is None or str(v).strip() == ""

    def _identity_key(row: dict) -> tuple[str, ...] | None:
        src = str(row.get("source", "")).strip()
        if src == "hedonic":
            return (
                "hedonic",
                str(row.get("data", "")).strip(),
                (str(row.get("split_strategy", "")).strip() or "random"),
                str(row.get("split_seed", "")).strip(),
                str(row.get("train_max_samples", "")).strip(),
                str(row.get("epochs", "")).strip(),
                str(row.get("batch_size", "")).strip(),
                str(row.get("lr", "")).strip(),
                str(row.get("weight_decay", "")).strip(),
            )
        if src == "train_ann":
            return (
                "train_ann",
                str(row.get("run_name", "")).strip(),
                str(row.get("data", "")).strip(),
                (str(row.get("split_strategy", "")).strip() or "random"),
                str(row.get("split_seed", "")).strip(),
            )
        if src == "train_fm":
            return (
                "train_fm",
                str(row.get("run_name", "")).strip(),
                str(row.get("data", "")).strip(),
                (str(row.get("split_strategy", "")).strip() or "random"),
                str(row.get("split_seed", "")).strip(),
            )
        if src == "train_deepfm":
            return (
                "train_deepfm",
                str(row.get("run_name", "")).strip(),
                str(row.get("data", "")).strip(),
                (str(row.get("split_strategy", "")).strip() or "random"),
                str(row.get("split_seed", "")).strip(),
            )
        if src == "train_tabnet":
            return (
                "train_tabnet",
                str(row.get("run_name", "")).strip(),
                str(row.get("data", "")).strip(),
                (str(row.get("split_strategy", "")).strip() or "random"),
                str(row.get("split_seed", "")).strip(),
            )
        return None

    merged_single: dict[tuple[str, ...], dict] = {}
    passthrough: list[dict] = []
    for row in rows_all:
        key = _identity_key(row)
        if key is None:
            passthrough.append(row)
            continue
        if key not in merged_single:
            merged_single[key] = dict(row)
            continue
        dst = merged_single[key]
        for k, v in row.items():
            if _is_blank(v):
                continue
            if k not in dst or _is_blank(dst.get(k)):
                dst[k] = v

    rows_all = passthrough + list(merged_single.values())

    # Build a stable header: common keys first, then the rest alphabetically.
    preferred = [
        "source",
        "stage",
        "run_group",
        "run",
        "run_name",
        "name",
        "config_id",
        "data",
        "feature_set",
        "seed",
        "split_seed",
        "split_strategy",
        "train_frac",
        "val_frac",
        "test_frac",
        "candidate_offset",
        "train_max_samples",
        "n_train_full",
        "n_train_used",
        "n_val",
        "n_test",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dropout",
        "embed_dim_cap",
        "cat_dropout",
        "n_d",
        "n_a",
        "n_steps",
        "gamma",
        "n_shared",
        "n_independent",
        "lambda_sparse",
        "hidden_dims",
        "factor_dim",
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
        "test_price_rmse",
        "seconds",
        "model",
        "notes",
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
