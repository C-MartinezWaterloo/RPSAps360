# Models

This repo compares multiple tabular models for predicting `TransactionPrice` (trained on `log1p(price)`), all using the same tensor format produced by `prepare_ann_tensors.py`.

Canonical results sheet (tracked on GitHub): `results_all.csv`

## Model index
- **Model 1: Hedonic regression (linear fixed effects)** → `models/01_hedonic/README.md`
- **Model 2: ANN (categorical embeddings + MLP)** → `models/02_ann/README.md`
- **Model 3: Factorization Machine (pairwise interactions)** → `models/03_fm/README.md`
- **Model 4: DeepFM (FM + MLP)** → `models/04_deepfm/README.md`
- **Model 5: TabNet (sequential feature selection)** → `models/05_tabnet/README.md`
- **Model 6: XGBoost (planned / external dependency)** → `models/06_xgboost/README.md`
