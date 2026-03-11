# Model 6 — XGBoost / LightGBM (external baseline)

XGBoost and LightGBM are strong baselines for tabular regression, but they require external compiled dependencies (`xgboost` / `lightgbm`).

This repo’s default environment targets a minimal dependency set (`numpy`, `torch`), so these are intended to be run in a separate Python environment.

## How to run

The runner script is:
- `models/06_xgboost/train_boosting.py`

It trains on the same tensors and split logic as the neural models and writes to:
- `xgb_runs.csv` (for XGBoost) or
- `lgbm_runs.csv` (for LightGBM)

These intermediate logs are gitignored and merged into the single canonical sheet via:
- `python export_results.py` → `results_all.csv`

Example (XGBoost, random split):
```bash
pip install xgboost
python models/06_xgboost/train_boosting.py --model xgboost --data ann_tensors_full.pt \
  --split-strategy random --seed 42 --run-name xgb_full_seed42 --out-csv xgb_runs.csv
python export_results.py
```

Example (LightGBM, time split):
```bash
pip install lightgbm
python models/06_xgboost/train_boosting.py --model lightgbm --data ann_tensors_full.pt \
  --split-strategy time --seed 42 --run-name lgbm_full_time_seed42 --out-csv lgbm_runs.csv
python export_results.py
```

## Note on categoricals

This repo’s tensors store categoricals as integer indices (for embeddings). To avoid huge one-hot matrices, the boosting script converts categoricals to compact numeric features using **frequency encoding** and **(K-fold) target mean encoding** computed on the training split only.
