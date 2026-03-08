# House Price Modeling Report (ANN + Hedonic Baseline)

This document summarizes what was built in this repo, what data we trained on, what experiments were run, and what the results mean.

## 1) What was built (end-to-end)

**Goal:** predict `TransactionPrice` using a small tabular model in PyTorch.

**Pipeline:**
1) **Convert the raw CSV into tensors** (`prepare_ann_tensors.py`)
   - Reads `cleaned_transaction_gtha-2.csv` (kept local / gitignored).
   - Builds:
     - `X_num` = numeric features (float32)
     - `X_cat` = categorical indices (int64) for embedding lookups
     - `y_log1p` = `log1p(TransactionPrice)` (float32)
   - Writes a single `.pt` file (kept local / gitignored).

2) **Train an ANN with categorical embeddings** (`train_ann.py`)
   - Uses a **fixed 70/15/15 split** (seed = 42).
   - Trains an MLP on `log1p(price)` using MSE loss (reported as RMSE).
   - Uses **early stopping**: keeps the epoch with best validation MSE.

3) **Run larger sweeps** (`sweep_ann.py`, `sweep_ann_deep_test.py`)
   - `sweep_ann.py` is **test-clean**: uses validation to pick the best config, evaluates test once.
   - `sweep_ann_deep_test.py` evaluates test every run (useful for exploration, but not “clean”).

4) **Train a hedonic regression baseline** (`hedonic_baseline.py`)
   - A linear model on `log1p(price)` with:
     - linear weights for numeric features
     - “fixed effects” for categoricals (implemented as 1D embeddings)

5) **Merge all experiment logs into one file** (`export_results.py`)
   - Writes the single canonical results sheet: `results_all.csv` (tracked for GitHub).

## 2) Data used (how many homes?)

All experiments used the same dataset size:
- Total rows (homes): **597,928**
- Split (seed=42):
  - Train: **418,549** (70%)
  - Validation: **89,689** (15%)
  - Test: **89,690** (15%)

The split is deterministic via `train_ann._make_splits()`.

## 3) Features used

Two feature sets exist because we started with your “minimal” list and later added a richer set that improved accuracy a lot:

### A) `feature_set=basic` (your original request)

Numeric:
- `LivingArea`, `LotSize`, `YearBuilt`, `DistanceToSubj`
- `Bath_Full`, `Bath_Half`, `Bed_Full`, `Bed_Half`

Categorical:
- `BuildingType`, `PropertyStyle`, `Condition`

### B) `feature_set=full` (stronger model)

Adds numeric:
- `gcode_Lat`, `gcode_Lon`, `TransactionYear`, `Quarter`, `DOM`
- `Parking_Count`, `LandValue`, `ReplacementCost`, `KitchenCount`, `EntranceCount`

Adds categorical:
- `FSA`, `gcode_City`, `Basement`, `Parking_Type`, `Ownership`, `PropertyUse`, `BRPS_Style`, `gcode_MatchStatus`

## 4) Target + metric (what we minimize)

We train and evaluate on:
- `y_log1p = log1p(TransactionPrice)`

Reported metric:
- `RMSE(log1p(TransactionPrice))`

Important:
- **Lower RMSE is better** (we minimize error; we do *not* maximize RMSE).

Intuition (rough):
- If `RMSE(log)` = `r`, a common “multiplicative error scale” approximation is `expm1(r)`.
  - Example: `r=0.143` → `expm1(r)≈0.154` (≈15% typical multiplicative error).

## 5) Results summary (what worked best)

All results live in `results_all.csv` (openable in Excel).

### Best models by feature set

**Best overall (ANN, full features):**
- Test RMSE(log1p): **0.1433** (≈15.4% multiplicative error scale)
- Config (from `results_all.csv`):
  - `hidden_dims=512,256,128,64`
  - `batch_size=512`
  - `lr=0.004`
  - `dropout=0.1`
  - `weight_decay=0.001`
  - `epochs=30` (best validation epoch ≈ 28)

**Best baseline on the same full features (hedonic regression):**
- Test RMSE(log1p): **0.2428** (≈27.5% multiplicative error scale)

**Best “basic feature set” ANN (test-clean sweep):**
- Test RMSE(log1p): **0.3425** (≈40.9% multiplicative error scale)

**Basic feature set hedonic baseline:**
- Test RMSE(log1p): **0.4722** (≈60.3% multiplicative error scale)

### What this says about “how well the ANN works”

- On the **basic** feature set, the ANN improves substantially over hedonic regression, but it does **not** get close to 0.2 RMSE.
- The big jump came from adding the **full** feature set:
  - Hedonic(full) improves a lot vs Hedonic(basic)
  - ANN(full) improves even more, reaching **0.143** test RMSE(log1p)

## 6) Hyperparameter conclusions (from the runs we logged)

### Full-feature ANN (the best-performing setup)

From the `train_ann` full-feature runs:
- **Depth helps**: the best run used a deeper MLP (`512 → 256 → 128 → 64`).
- **Dropout helps**: `dropout=0.0` was meaningfully worse than `0.05–0.1`.
- **Learning rate sweet spot**: `lr=0.003–0.004` was best; `0.005` was slightly worse.
- **Weight decay**: `5e-4–1e-3` performed well (and is safer than 0 when models get larger).
- **Batch size**: `512` worked best for the top full runs.
- **Embedding cap**: increasing `embed_dim_cap` above 64 did not improve the best run here.

### Basic-feature ANN (sweep results)

From `sweep_clean` (test-clean):
- Larger/deeper networks helped, but the best clean test RMSE plateaued around **0.34**.

## 7) Repro commands (exactly what to run)

1) Build tensors (full feature set):
```bash
python prepare_ann_tensors.py --csv cleaned_transaction_gtha-2.csv --out ann_tensors_full.pt --feature-set full
```

2) Train the best full-feature ANN we found:
```bash
python train_ann.py --data ann_tensors_full.pt --epochs 30 --batch-size 512 --lr 0.004 --weight-decay 0.001 \\
  --hidden-dims 512,256,128,64 --dropout 0.1 --embed-dim-cap 64 --seed 42
```

3) Run hedonic baseline (full features):
```bash
python hedonic_baseline.py --data ann_tensors_full.pt --out-csv hedonic_results.csv
```

4) Export merged results:
```bash
python export_results.py
```

## 8) Caveats / evaluation notes

- **Test set hygiene:** `sweep_ann.py` is “test-clean”. `sweep_ann_deep_test.py` reports test for every run and should be treated as exploratory (hyperparameter selection can leak through repeated test evaluation).
- **Standardization + vocab leakage:** `prepare_ann_tensors.py` computes numeric mean/std and categorical vocab using the whole dataset (not train-only). This is simpler, but slightly optimistic. For stricter evaluation, compute these using training data only and apply to val/test.

## 9) Using year to predict future prices (time-based split)

If the intent is “learn inflation / market trend” and then predict later years, a random split is **not** the right evaluation: the model will see all years in train/val/test and you are mostly measuring interpolation.

To support a forecast-style evaluation, we added:
- `prepare_ann_tensors.py --feature-set basic_time` (minimal features + `TransactionYear`/`Quarter`)
- `train_ann.py --split-strategy time` / `hedonic_baseline.py --split-strategy time` (train on earlier transactions, test on later ones)

**What happened to accuracy with a time split (seed=42, 70/15/15 by time-order):**
- ANN (full features): **test RMSE(log1p) = 0.2609**
- Hedonic baseline (full features): **test RMSE(log1p) = 0.5815**
- ANN (basic_time = minimal + year/quarter): **test RMSE(log1p) = 0.4079**
- Hedonic baseline (basic_time): **test RMSE(log1p) = 1.4119**

Interpretation:
- Including `TransactionYear` helps the model represent nominal price changes over time, but **forecasting later years is much harder** than a random split.
- The **full** feature set still matters a lot (location + extra attributes). “Year alone” does not get close to the full-feature ANN performance.
