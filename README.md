# RPSAps360 – House Price Modeling (Neural Tabular Models)

This repo prepares a housing transaction dataset into PyTorch tensors, trains multiple tabular models, and logs results to a single CSV (`results_all.csv`).

For a detailed write-up of the pipeline and the results, see `REPORT.md`.
For a PDF-style write-up with embedded graphs, see `report.tex`.
For the **APS360 final submission** report, see `final_report.tex`.
For a **graphs-only appendix** (lots of quant + qual figures), see `graphs_report.tex`.
Model-specific PDF-style write-ups:
- Hedonic baseline: `hedonic_baseline_report.tex`
- Primary model (ANN): `ann_primary_report.tex`
- DeepFM: `deepfm_report.tex`

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
- `train_fttransformer.py` – trains an FT-Transformer model (Transformer for tabular data)
- `train_tabtransformer.py` – trains a TabTransformer model (attention over categorical embeddings)
- `eval_suite.py` – reruns best configs across multiple seeds/splits (random + time split)
- `feature_importance.py` – permutation feature importance (ANN vs hedonic)
- `make_report_graphs.py` – generates an offline HTML dashboard of plots from `results_all.csv`
- `open_graphs.py` – opens the HTML dashboard in your browser
- `graphs.html` – double-click convenience redirect to `plots/report_graphs.html`
- `make_overleaf_bundle.py` – creates an Overleaf-ready zip (compile PDFs without local TeX)
- `export_results.py` – merges all run logs into **one** file: `results_all.csv`
- `results_all.csv` – **canonical** results file for GitHub
- `report.tex` – LaTeX report (pgfplots graphs + tables)
- `hedonic_baseline_report.tex` – hedonic baseline write-up (diagram + learning curves + qualitative graphs)
- `ann_primary_report.tex` – primary model write-up (ANN diagram + complexity + quantitative + qualitative results)
- `deepfm_report.tex` – DeepFM write-up (architecture diagram + sweeps + robustness + qualitative graphs)
- `models/` – short docs per model (Hedonic, ANN, FM, DeepFM, TabNet, planned XGBoost)

## Quick start
0) Install Python deps:
```bash
python -m pip install -r requirements.txt
```

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

## View graphs (no LaTeX)
- Fastest: double-click `graphs.html` (opens `plots/report_graphs.html`)
- Or run: `python open_graphs.py`
- If the HTML doesn’t exist yet, generate it with:
  - `python make_report_graphs.py --results results_all.csv --out plots/report_graphs.html`

## Compile PDFs (Overleaf)
This environment may not have `pdflatex` installed. To compile `final_report.tex` / `graphs_report.tex`:
```bash
python make_overleaf_bundle.py
```
Upload the generated `overleaf_bundle.zip` to Overleaf and set the main file to the report you want.

## Notes
- Metrics include both:
  - `*_rmse` on `log1p(TransactionPrice)` (RMSLE-like; lower is better)
  - `*_acc_pct` in price space, derived from `MdAPE` (median absolute percentage error)
- Large files (dataset, `.pt` tensors, Excel/doc files, intermediate sweep logs) are gitignored; only `results_all.csv` is intended to be committed.
