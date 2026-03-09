# Model 6 — XGBoost (planned)

XGBoost is a strong baseline for tabular regression, but it requires external compiled dependencies (`xgboost`, usually `scikit-learn` tooling too).

This environment currently targets a minimal dependency set (`numpy`, `torch`), so we keep XGBoost as a planned extension.

If you want this added later:
1) Install `xgboost` in your Python environment
2) Add a script to train + sweep XGBoost and log into `results_all.csv`
