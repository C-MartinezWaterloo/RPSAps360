"""Linear hedonic pricing model built from the cleaned transaction data."""

import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CSV_PATH = "cleaned_transaction_gtha.csv"
TARGET_COL = "TransactionPrice"

# ---- knobs you can tune ----
LOW_CARD_MAX_UNIQUE = 200     # one-hot if column has <= this many unique values
HASH_DIM = 2**18              # hashed feature dimension for high-card columns (262,144)
HASH_SIGNED = False           # False => nonnegative hashed features
TEST_SIZE = 0.2
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0
# ----------------------------


def build_feature_matrix(df: pd.DataFrame) -> sparse.csr_matrix:
    """Create a sparse feature matrix with numeric, one-hot, and hashed columns."""
    # Convert TransactionDate to numeric parts (prevents huge one-hot on full dates)
    if "TransactionDate" in df.columns:
        dt = pd.to_datetime(df["TransactionDate"], errors="coerce")
        df = df.copy()
        df["TransactionDate_year"] = dt.dt.year
        df["TransactionDate_month"] = dt.dt.month
        df["TransactionDate_day"] = dt.dt.day
        df = df.drop(columns=["TransactionDate"])

    # Identify numeric vs non-numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # Numeric matrix (sparse)
    X_num = df[numeric_cols].copy()
    for col in numeric_cols:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    X_num = X_num.fillna(X_num.median(numeric_only=True))
    X_num_mat = sparse.csr_matrix(X_num.to_numpy(dtype=np.float32))

    # Split categoricals into low-card one-hot vs high-card hashed
    low_card, high_card = [], []
    for col in cat_cols:
        nunq = df[col].replace("", np.nan).nunique(dropna=True)
        if nunq <= LOW_CARD_MAX_UNIQUE:
            low_card.append(col)
        else:
            high_card.append(col)

    # One-hot (low-card)
    if low_card:
        low_df = df[low_card].astype("string").fillna("__NA__")
        X_low = pd.get_dummies(low_df, dummy_na=False, dtype="uint8")
        X_low_mat = sparse.csr_matrix(X_low.values, dtype=np.float32)
    else:
        X_low_mat = sparse.csr_matrix((len(df), 0), dtype=np.float32)

    # Hashing (high-card)
    if high_card:
        high_df = df[high_card].astype("string").fillna("__NA__")
        tokens = []
        for i in range(len(high_df)):
            row = high_df.iloc[i]
            tokens.append([f"{col}={row[col]}" for col in high_card])

        hasher = FeatureHasher(
            n_features=HASH_DIM,
            input_type="string",
            alternate_sign=HASH_SIGNED,
        )
        X_hash_mat = hasher.transform(tokens).astype(np.float32)
    else:
        X_hash_mat = sparse.csr_matrix((len(df), 0), dtype=np.float32)

    return sparse.hstack([X_num_mat, X_low_mat, X_hash_mat], format="csr", dtype=np.float32)


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in {CSV_PATH}.")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    X_raw = df.drop(columns=[TARGET_COL])

    X_all = build_feature_matrix(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print("Linear hedonic pricing model results:")
    print(f"  Samples: {X_all.shape[0]} | Features: {X_all.shape[1]}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R^2: {r2:.4f}")


if __name__ == "__main__":
    main()
