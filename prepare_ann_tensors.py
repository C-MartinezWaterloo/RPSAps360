#!/usr/bin/env python3
"""
Prepare tensors for an ANN from cleaned transaction CSV.

Outputs a single torch file containing:
  - X_num: float32 tensor [N, n_numeric] (standardized)
  - X_cat: int64 tensor [N, n_categorical] (category indices)
  - y: float32 tensor [N, 1] (raw TransactionPrice)
  - y_log1p: float32 tensor [N, 1] (log1p(TransactionPrice))
  - metadata: column lists, category maps, numeric stats

This avoids one-hotting high-cardinality categoricals (e.g. PropertyStyle),
which would create an impractically wide dense matrix for an ANN.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


TARGET_COL = "TransactionPrice"

# --------------------------
# Feature sets
# --------------------------
# "basic" matches what we started with (small feature set).
NUMERIC_COLS_BASIC = [
    "LivingArea",
    "LotSize",
    "YearBuilt",
    "DistanceToSubj",
    "Bath_Full",
    "Bath_Half",
    "Bed_Full",
    "Bed_Half",
]
CAT_COLS_BASIC = [
    "BuildingType",
    "PropertyStyle",
    "Condition",
]

# "full" adds location + time + a few extra property attributes.
# This should improve accuracy substantially compared to "basic".
NUMERIC_COLS_FULL = [
    *NUMERIC_COLS_BASIC,
    "gcode_Lat",
    "gcode_Lon",
    "TransactionYear",
    "Quarter",
    "DOM",
    "Parking_Count",
    "LandValue",
    "ReplacementCost",
    "KitchenCount",
    "EntranceCount",
]
CAT_COLS_FULL = [
    *CAT_COLS_BASIC,
    "FSA",
    "gcode_City",
    "Basement",
    "Parking_Type",
    "Ownership",
    "PropertyUse",
    "BRPS_Style",
    "gcode_MatchStatus",
]


def _clean_str(value: Any) -> str:
    """Convert any value to a stripped string (safe for None)."""
    if value is None:
        return ""
    return str(value).strip()


def _is_missing_token(s: str) -> bool:
    """Return True if a string should be treated as missing."""
    if s == "":
        return True
    sl = s.lower()
    return sl in {"na", "n/a", "nan", "null", "none"} or s in {"-999", "-999.0"}


def _parse_float(value: Any) -> float | None:
    """Parse a float; return None if missing/invalid."""
    s = _clean_str(value)
    if _is_missing_token(s):
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    if x == -999.0:
        return None
    return x


@dataclass
class OnlineStats:
    """
    Streaming mean/std via Welford's algorithm.

    Why: the CSV is large, so we avoid loading all values into memory just to
    compute mean/std for standardization.
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def finalize(self) -> tuple[float, float]:
        if self.n <= 1:
            return (self.mean, 0.0)
        var = self.m2 / (self.n - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        return (self.mean, std)


def _build_category_maps(
    cat_counts: dict[str, Counter[str]],
    max_categories: int | None,
) -> dict[str, dict[str, int]]:
    """
    Build a mapping {column: {category_string: integer_index}}.

    - "__NA__" always exists (used for missing values)
    - If `max_categories` is set, categories are truncated by frequency and
      "__OTHER__" is added for anything unseen/trimmed.
    """

    maps: dict[str, dict[str, int]] = {}
    for col, counter in cat_counts.items():
        counter = counter.copy()
        counter.setdefault("__NA__", 0)

        items = list(counter.items())
        items.sort(key=lambda kv: (-kv[1], kv[0]))

        keep: list[str]
        if max_categories is None:
            keep = [k for k, _ in items]
        else:
            keep = [k for k, _ in items[: max(1, max_categories)]]

        if "__NA__" not in keep:
            keep.insert(0, "__NA__")

        # If truncating, reserve __OTHER__ for unseen/trimmed categories.
        if max_categories is not None and "__OTHER__" not in keep:
            keep.append("__OTHER__")

        col_map: dict[str, int] = {cat: i for i, cat in enumerate(keep)}
        maps[col] = col_map
    return maps


def _resolve_required_columns(header: list[str]) -> None:
    missing = [c for c in (NUMERIC_COLS + CAT_COLS + [TARGET_COL]) if c not in header]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="cleaned_transaction_gtha-2.csv")
    parser.add_argument("--out", default="ann_tensors.pt")
    parser.add_argument(
        "--feature-set",
        choices=["basic", "full"],
        default="basic",
        help="Which set of columns to use (default: basic).",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=None,
        help="Cap categories per categorical column; extra values map to __OTHER__.",
    )
    parser.add_argument(
        "--drop-missing-target",
        action="store_true",
        default=True,
        help="Drop rows with missing/non-positive TransactionPrice (default: on).",
    )
    parser.add_argument(
        "--keep-missing-target",
        dest="drop_missing_target",
        action="store_false",
        help="Keep rows even if TransactionPrice is missing (y will be 0).",
    )
    args = parser.parse_args()

    if args.feature_set == "basic":
        numeric_cols = list(NUMERIC_COLS_BASIC)
        cat_cols = list(CAT_COLS_BASIC)
    else:
        numeric_cols = list(NUMERIC_COLS_FULL)
        cat_cols = list(CAT_COLS_FULL)

    # PASS 1 (stream over CSV):
    #   - count category frequencies (to build vocab -> integer index)
    #   - compute mean/std for numeric features (for standardization)
    #   - count how many rows we will keep (so we can pre-allocate arrays)
    num_stats = {c: OnlineStats() for c in numeric_cols}
    cat_counts = {c: Counter() for c in cat_cols}
    n_total = 0
    n_keep = 0
    n_dropped_target = 0

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        missing = [c for c in (numeric_cols + cat_cols + [TARGET_COL]) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns for feature-set='{args.feature_set}': {missing}")

        for row in reader:
            n_total += 1

            y_val = _parse_float(row.get(TARGET_COL))
            keep_row = True
            if args.drop_missing_target:
                keep_row = y_val is not None and y_val > 0
            if not keep_row:
                n_dropped_target += 1
                continue

            n_keep += 1

            for col in numeric_cols:
                x = _parse_float(row.get(col))
                if x is not None:
                    num_stats[col].update(x)

            for col in cat_cols:
                # Normalize categoricals to reduce accidental duplicates
                # (e.g., "Residential" vs "RESIDENTIAL").
                v = _clean_str(row.get(col)).upper()
                if _is_missing_token(v):
                    v = "__NA__"
                cat_counts[col][v] += 1

    if n_keep == 0:
        raise ValueError("No rows kept; check target filtering and input data.")

    num_final = {c: num_stats[c].finalize() for c in numeric_cols}
    cat_maps = _build_category_maps(cat_counts, args.max_categories)

    # PASS 2 (stream again):
    #   - build dense numeric feature matrix (standardized)
    #   - build categorical index matrix (for embeddings)
    #   - build the target vector
    X_num = np.zeros((n_keep, len(numeric_cols)), dtype=np.float32)
    X_cat = np.zeros((n_keep, len(cat_cols)), dtype=np.int64)
    y = np.zeros((n_keep,), dtype=np.float32)

    row_i = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_val = _parse_float(row.get(TARGET_COL))
            keep_row = True
            if args.drop_missing_target:
                keep_row = y_val is not None and y_val > 0
            if not keep_row:
                continue

            if y_val is None:
                y_val = 0.0
            y[row_i] = float(y_val)

            for j, col in enumerate(numeric_cols):
                x = _parse_float(row.get(col))
                mean, std = num_final[col]
                if x is None:
                    # Missing numeric -> impute with training mean.
                    x = mean
                if std > 0:
                    # Standardize: (x - mean) / std
                    x = (x - mean) / std
                else:
                    # If std is 0, the feature is constant -> all zeros after standardization.
                    x = 0.0
                X_num[row_i, j] = float(x)

            for j, col in enumerate(cat_cols):
                v = _clean_str(row.get(col)).upper()
                if _is_missing_token(v):
                    v = "__NA__"
                col_map = cat_maps[col]
                idx = col_map.get(v)
                if idx is None:
                    # Unseen category at inference time -> map to __OTHER__ (or __NA__ if not present).
                    idx = col_map.get("__OTHER__", col_map.get("__NA__", 0))
                X_cat[row_i, j] = int(idx)

            row_i += 1

    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    y_log1p_t = torch.log1p(y_t)

    payload = {
        "X_num": torch.tensor(X_num, dtype=torch.float32),
        "X_cat": torch.tensor(X_cat, dtype=torch.int64),
        "y": y_t,
        "y_log1p": y_log1p_t,
        "numeric_cols": list(numeric_cols),
        "cat_cols": list(cat_cols),
        "cat_maps": cat_maps,
        "num_stats": {c: {"mean": float(num_final[c][0]), "std": float(num_final[c][1])} for c in numeric_cols},
        "source_csv": args.csv,
        "feature_set": args.feature_set,
        "n_total": n_total,
        "n_kept": n_keep,
        "n_dropped_target": n_dropped_target,
    }

    torch.save(payload, args.out)

    print("Saved:", args.out)
    print(f"Rows: kept={n_keep} / total={n_total} (dropped_target={n_dropped_target})")
    print("X_num:", payload["X_num"].shape)
    print("X_cat:", payload["X_cat"].shape)
    print("y:", payload["y"].shape)
    for col in cat_cols:
        print(f"{col}: {len(cat_maps[col])} categories")


if __name__ == "__main__":
    main()
