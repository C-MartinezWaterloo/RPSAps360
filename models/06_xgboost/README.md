# Model 6 — XGBoost / LightGBM (planned)

XGBoost and LightGBM are strong baselines for tabular regression, but they require external compiled dependencies (`xgboost` / `lightgbm`, and often `scikit-learn` tooling too).

This environment currently targets a minimal dependency set (`numpy`, `torch`), so we keep XGBoost as a planned extension.

If you want this added later:
1) Install `xgboost` and/or `lightgbm` in your Python environment
2) Train + sweep on the same split logic (random + time split)
3) Log results into `results_all.csv` using the same schema (`*_rmse`, `*_acc_pct`, and `*_last` metrics)

Practical workflow:
- Run boosting models in a separate environment (where compiled deps work), write `xgb_runs.csv`/`lgbm_runs.csv`, then ingest them via `export_results.py`.
