# Model 1 — Hedonic regression (baseline)

**Script:** `hedonic_baseline.py`

## What it is
A classic housing baseline:
- linear weights on numeric features
- per-category “fixed effects” (implemented as `Embedding(size, 1)`)

## How to run
```bash
python hedonic_baseline.py --data ann_tensors_full.pt --out-csv hedonic_results.csv
python export_results.py
```

## Metrics in `results_all.csv`
- `test_rmse`: RMSE in `log1p(price)` space (lower is better)
- `test_mdape`: median APE in price space (lower is better)
- `test_acc_pct`: `max(0, 100*(1 - test_mdape))` (higher is better)

