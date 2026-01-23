# APS360.py
# Build a single PyTorch (sparse) tensor from a mixed-type CSV:
# - numeric columns -> numeric features
# - low-cardinality categoricals -> one-hot
# - high-cardinality categoricals/text -> hashing trick (fixed width)
# Also builds y from TransactionPrice (if present).

import pandas as pd
import numpy as np
import torch

from scipy import sparse
from sklearn.feature_extraction import FeatureHasher

CSV_PATH = "cleaned_transaction_gtha.csv"
TARGET_COL = "TransactionPrice"

# Columns to use for modeling. Missing columns are reported and skipped.
MODEL_COLUMNS = [
    "LivingArea",
    "LotSize",
    "YearBuilt",
    "KitchenCount",
    "EntranceCount",
    "PropertyStyle",
    "BuildingType",
    "BRPS_Style",
    "Bath_Full",
    "Bath_Half",
    "Bed_Full",
    "Bed_Half",
    "Condition",
    "Basement",
    "Parking_Count",
    "DistanceToSubj",
]

# ---- knobs you can tune ----
LOW_CARD_MAX_UNIQUE = 200     # one-hot if column has <= this many unique values
HASH_DIM = 2**18              # hashed feature dimension for high-card columns (262,144)
HASH_SIGNED = False           # False => nonnegative hashed features
# ----------------------------


def main():
    # 1) Load
    df = pd.read_csv(CSV_PATH)

    # 1b) Restrict to requested modeling columns when present
    available_cols = [c for c in MODEL_COLUMNS if c in df.columns]
    missing_cols = [c for c in MODEL_COLUMNS if c not in df.columns]
    if missing_cols:
        print("Warning: missing columns skipped:")
        print("  ", ", ".join(missing_cols))
    if available_cols:
        df = df[available_cols + ([TARGET_COL] if TARGET_COL in df.columns else [])]

    # 2) Separate target
    if TARGET_COL in df.columns:
        y_np = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        y = torch.tensor(y_np).view(-1, 1)
        X_raw = df.drop(columns=[TARGET_COL])
    else:
        y = None
        X_raw = df

    # 3) Convert TransactionDate to numeric parts (prevents huge one-hot on full dates)
    if "TransactionDate" in X_raw.columns:
        dt = pd.to_datetime(X_raw["TransactionDate"], errors="coerce")
        X_raw["TransactionDate_year"] = dt.dt.year
        X_raw["TransactionDate_month"] = dt.dt.month
        X_raw["TransactionDate_day"] = dt.dt.day
        X_raw = X_raw.drop(columns=["TransactionDate"])

    # 4) Identify numeric vs non-numeric
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_raw.columns if c not in numeric_cols]

    # 5) Numeric matrix (sparse)
    X_num = X_raw[numeric_cols].copy()
    for c in numeric_cols:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
    X_num = X_num.fillna(X_num.median(numeric_only=True))
    X_num_mat = sparse.csr_matrix(X_num.to_numpy(dtype=np.float32))

    # 6) Split categoricals into low-card one-hot vs high-card hashed
    low_card, high_card = [], []
    for c in cat_cols:
        # treat empty strings as missing for counting uniques
        nunq = X_raw[c].replace("", np.nan).nunique(dropna=True)
        if nunq <= LOW_CARD_MAX_UNIQUE:
            low_card.append(c)
        else:
            high_card.append(c)

    # 6a) One-hot (LOW CARD) -> scipy sparse
    if low_card:
        low_df = X_raw[low_card].astype("string").fillna("__NA__")
        X_low = pd.get_dummies(
            low_df,
            dummy_na=False,
            dtype="uint8"   # IMPORTANT: avoids BooleanDtype / numpy dtype errors
        )
        X_low_mat = sparse.csr_matrix(X_low.values, dtype=np.float32)
    else:
        X_low_mat = sparse.csr_matrix((len(X_raw), 0), dtype=np.float32)

    # 6b) Hashing (HIGH CARD) -> scipy sparse fixed width
    if high_card:
        high_df = X_raw[high_card].astype("string").fillna("__NA__")

        # Build tokens per row so hashing distinguishes columns:
        # ["col=value", "col2=value2", ...]
        tokens = []
        for i in range(len(high_df)):
            row = high_df.iloc[i]
            tokens.append([f"{col}={row[col]}" for col in high_card])

        hasher = FeatureHasher(
            n_features=HASH_DIM,
            input_type="string",
            alternate_sign=HASH_SIGNED
        )
        X_hash_mat = hasher.transform(tokens).astype(np.float32)
    else:
        X_hash_mat = sparse.csr_matrix((len(X_raw), 0), dtype=np.float32)

    # 7) Combine into one sparse matrix
    X_all = sparse.hstack([X_num_mat, X_low_mat, X_hash_mat], format="csr", dtype=np.float32)

    # 8) Convert scipy sparse -> PyTorch sparse COO tensor
    X_coo = X_all.tocoo()
    indices = torch.tensor(np.vstack([X_coo.row, X_coo.col]), dtype=torch.int64)
    values = torch.tensor(X_coo.data, dtype=torch.float32)
    X = torch.sparse_coo_tensor(indices, values, size=X_coo.shape).coalesce()

    print("Encoded feature tensor X (sparse):")
    print("  shape:", tuple(X.shape))
    print("  nnz:", X._nnz())

    if y is not None:
        print("Target tensor y:")
        print("  shape:", tuple(y.shape))
        print("  first 5:", y[:5].view(-1).tolist())

    # Optional: save tensors (comment out if you don't want files)
    # torch.save(X, "X_encoded_sparse.pt")
    # if y is not None:
    #     torch.save(y, "y_transaction_price.pt")


if __name__ == "__main__":
    main()
