# RPSAps360 – House Price Modeling (ANN + Baselines)

This repo prepares a housing transaction dataset into PyTorch tensors, trains multiple tabular models, and logs results to a single CSV (`results_all.csv`).

For a detailed write-up of the pipeline and the results, see `REPORT.md`.
For a PDF-style write-up with embedded graphs, see `report.tex`.

## Files
- `prepare_ann_tensors.py` – reads the raw CSV and writes `.pt` tensors
- `train_ann.py` – trains a single ANN run (70/15/15 split) and can append a row to `ann_runs.csv`
- `sweep_ann.py` – “test-clean” sweep (select by validation; test evaluated once)
- `sweep_ann_deep_test.py` – deeper/longer sweep that reports test for every run (test-focused)
- `hedonic_baseline.py` – linear “hedonic regression” baseline (fixed effects for categoricals)
- `train_fm.py` – trains a Factorization Machine (FM) baseline (pairwise interactions)
- `sweep_fm.py` – runs a randomized FM sweep (logs train/val/test + overfitting via `*_last`)
- `train_deepfm.py` – trains a DeepFM model (FM + deep MLP)
- `sweep_deepfm.py` – runs a randomized DeepFM sweep
- `train_tabnet.py` – trains a TabNet model (sequential feature selection)
- `sweep_tabnet.py` – runs a randomized TabNet sweep
- `eval_suite.py` – reruns best configs across multiple seeds/splits (random + time split)
- `feature_importance.py` – permutation feature importance (ANN vs hedonic)
- `make_report_graphs.py` – generates an offline HTML dashboard of plots from `results_all.csv`
- `export_results.py` – merges all run logs into **one** file: `results_all.csv`
- `results_all.csv` – **canonical** results file for GitHub
- `report.tex` – LaTeX report (pgfplots graphs + tables)
- `models/` – short docs per model (Hedonic, ANN, FM, DeepFM, TabNet, planned XGBoost)

## Quick start
1) Put the dataset CSV in the repo root (it is gitignored):
   - `cleaned_transaction_gtha-2.csv`

2) Build tensors (recommended “full” feature set):
```bash
python prepare_ann_tensors.py --csv cleaned_transaction_gtha-2.csv --out ann_tensors_full.pt --feature-set full
```

If you want a minimal feature set *plus* transaction time (year/quarter), use:
```bash
python prepare_ann_tensors.py --csv cleaned_transaction_gtha-2.csv --out ann_tensors_basic_time.pt --feature-set basic_time
```

3) Train a single ANN run:
```bash
python train_ann.py --data ann_tensors_full.pt --epochs 30 --batch-size 512 --lr 0.004 --weight-decay 0.001 \
  --hidden-dims 512,256,128,64 --dropout 0.1 --embed-dim-cap 64 --seed 42 \
  --out-csv ann_runs.csv --run-name my_run
```

To evaluate “future prediction” (train on earlier years, test on later years), add:
`--split-strategy time`

4) Export/merge all results into one CSV:
```bash
python export_results.py
```
This overwrites `results_all.csv` with the merged output.

## Notes
- Metrics include both:
  - `*_rmse` on `log1p(TransactionPrice)` (RMSLE-like; lower is better)
  - `*_acc_pct` in price space, derived from `MdAPE` (median absolute percentage error)
- Large files (dataset, `.pt` tensors, Excel/doc files, intermediate sweep logs) are gitignored; only `results_all.csv` is intended to be committed.
