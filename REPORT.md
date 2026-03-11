# House Price Modeling Report (Hedonic + ANN + FM + DeepFM + TabNet)

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

5) **Train a Factorization Machine (FM)** (`train_fm.py`, `sweep_fm.py`)
   - A lightweight tabular model that adds **pairwise interactions** via low-rank factors.
   - Useful “middle ground” between linear hedonic regression and a deep MLP.

6) **Train a DeepFM model** (`train_deepfm.py`, `sweep_deepfm.py`)
   - Combines linear terms + FM interactions + a deep MLP.

7) **Train a TabNet model** (`train_tabnet.py`, `sweep_tabnet.py`)
   - Sequential feature selection with sparse masks (sparsemax).

8) **Merge all experiment logs into one file** (`export_results.py`)
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

Additional metric (for your “accuracy %” request):
- `MdAPE` (median absolute percentage error) in **price space**
- `accuracy_pct = max(0, 100*(1 - MdAPE))`

Important:
- **Lower RMSE is better** (we minimize error; we do *not* maximize RMSE).

Intuition (rough):
- If `RMSE(log)` = `r`, a common “multiplicative error scale” approximation is `expm1(r)`.
  - Example: `r=0.143` → `expm1(r)≈0.154` (≈15% typical multiplicative error).

## 5) Results summary (what worked best)

All results live in `results_all.csv` (openable in Excel).

### A) Full-feature comparison (MdAPE-based accuracy %)

These are the best runs we logged with `test_acc_pct` available (note: many sweeps cap training rows to 50,000 for speed; full-train runs are explicitly labeled):

| Model | Feature set | Train rows | Test accuracy % | Test RMSE(log) | Notes |
|---|---|---:|---:|---:|---|
| Hedonic (linear fixed effects) | full | 418,549 | 87.74 | 0.2428 | Full-train baseline |
| ANN (full train, best single run) | full | 418,549 | 92.62 | 0.1433 | `512→256→128→64`, `dropout=0.1` |
| DeepFM (full train, single run) | full | 418,549 | 91.00 | 0.1791 | `k=16`, `2048→1024→512`, `dropout=0.05` |
| ANN sweep (deep_test, 100 runs) | full | 50,000 | 88.02 | 0.2564 | Best config: `huge3` (2048→1024→512) |
| FM sweep (100 runs) | full | 50,000 | 88.33 | 0.4056 | Higher median accuracy, but much worse tail error (RMSE) |
| DeepFM sweep (200 runs) | full | 50,000 | 89.70 | 0.2851 | Best config: `huge` + `k=16` + deep MLP (2048→1024→512) |
| TabNet sweep (100 runs) | full | 50,000 | 78.46 | 0.3948 | Best config: `d=16,a=16,steps=3,gamma=1.3` |

Interpretation:
- **Accuracy% (MdAPE)** is robust to outliers; **RMSE(log)** is sensitive to the tail.
- FM/DeepFM can look strong on MdAPE while still having much worse RMSE if a small fraction of homes are predicted badly.

Additional note (full-train DeepFM stability):
- Across 3 random-split seeds (0/1/42), the DeepFM full-train config averaged **test acc% ≈ 91.22 ± 0.51** (see `train_deepfm` rows in `results_all.csv`).

### B) Best overall RMSE (full training)

The best run we logged on `feature_set=full` by **test RMSE(log)** (lower is better):
- ANN (full train): **test RMSE(log1p) ≈ 0.1433**, **test accuracy% ≈ 92.62%** (MdAPE ≈ 7.38%)

This is substantially better than the FM/ANN 50k sweeps above; those sweeps cap training rows for speed.

### Best models by feature set

**Best overall (ANN, full features):**
- Test RMSE(log1p): **0.1433** (≈15.4% multiplicative error scale)
- Test accuracy% (MdAPE): **92.62%** (MdAPE ≈ 7.38%)
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

## 6.5) Robustness checks (multiple seeds + splits)

To verify the ANN’s advantage is not due to a “lucky” random partition, we reran the best full-feature ANN and the hedonic baseline across multiple **random split seeds** and **model seeds**, and also on a **time-based split** (train earlier years → test later years). These runs are logged under `source=eval_suite` in `results_all.csv`.

**Random split (21 runs; multiple split seeds + model seeds):**
- ANN best config (`hidden_dims=512,256,128,64`): **test RMSE ≈ 0.1420 ± 0.0036**, **test acc% ≈ 92.64 ± 0.31**
- Hedonic baseline: **test RMSE ≈ 0.2446 ± 0.0030**, **test acc% ≈ 87.76 ± 0.10**

**Time split (9 runs; split_seed ∈ {42,7}):**
- ANN best config: **test RMSE ≈ 0.3146 ± 0.0365**, **test acc% ≈ 82.78 ± 2.35**
- Hedonic baseline: **test RMSE ≈ 0.5947 ± 0.0162**, **test acc% ≈ 69.35 ± 1.09**

Notes:
- The **time split** is much more sensitive to model initialization than the random split. For the ANN across these seeds, **best** was **RMSE=0.2609 / acc=85.78%** (seed=42) and **worst** was **RMSE=0.3727 / acc=77.27%** (seed=2).

Interpretation:
- The ANN remains clearly better than hedonic regression across multiple random partitions.
- Performance drops significantly under a time-based split (as expected), but the ANN still generalizes much better than the hedonic baseline.

## 6.6) Feature importance (quick permutation analysis)

To better understand *why* the ANN improves over the hedonic model, we ran a simple permutation importance analysis on a validation subset (20,000 rows; random split seed=42). Importance is measured as the increase in validation MSE when a single feature is randomly permuted.

**Top drivers for both models:**
- `LivingArea` (largest impact)
- `TransactionYear` (captures market trend / price level)
- Location features (`FSA`, `gcode_City`, `gcode_Lat`, `gcode_Lon`)

Why the ANN can beat hedonic regression:
- The ANN can learn **nonlinear effects** (e.g., diminishing returns to size, different year trends by location) and **interactions** (size × neighborhood, style × city, etc.), while the hedonic model is constrained to be linear in the standardized numeric features plus fixed categorical offsets.

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

**What happened to accuracy with a time split (70/15/15 by time-order):**
- ANN (full features; `eval_suite`, 9 runs across multiple model seeds and `split_seed ∈ {42,7}`):
  - **test RMSE(log1p) ≈ 0.3146 ± 0.0365**
  - **test accuracy% ≈ 82.78 ± 2.35**
  - Best run in these checks: **RMSE=0.2609 / acc=85.78%** (seed=42, split_seed=42)
- Hedonic baseline (full features; `eval_suite`, 9 runs):
  - **test RMSE(log1p) ≈ 0.5947 ± 0.0162**
  - **test accuracy% ≈ 69.35 ± 1.09**
- DeepFM (full features; full-train, 3 seeds 0/1/42):
  - **test RMSE(log1p) ≈ 0.3533 ± 0.0107**
  - **test accuracy% ≈ 75.74 ± 1.51**
- ANN (basic_time = minimal + year/quarter): **test RMSE(log1p) = 0.4079**
- Hedonic baseline (basic_time): **test RMSE(log1p) = 1.4119**

Interpretation:
- Including `TransactionYear` helps the model represent nominal price changes over time, but **forecasting later years is much harder** than a random split.
- The **full** feature set still matters a lot (location + extra attributes). “Year alone” does not get close to the full-feature ANN performance.

## 10) External baselines (XGBoost / LightGBM)

XGBoost/LightGBM are commonly very strong on tabular data and are worth comparing against the neural nets here. The default environment for this repo is intentionally minimal and does not include compiled boosting dependencies, so these are meant to be run in a separate Python environment.

When you’re ready to add them:
- Use `models/06_xgboost/train_boosting.py` to train either XGBoost or LightGBM on the same tensors + split logic.
- It writes `xgb_runs.csv` / `lgbm_runs.csv` (gitignored), which are merged into `results_all.csv` by running `python export_results.py`.
- To avoid huge one-hot matrices, the script uses compact categorical encodings (frequency + K-fold target mean encoding computed on the training split only).
