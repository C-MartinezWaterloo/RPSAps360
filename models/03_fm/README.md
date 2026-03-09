# Model 3 — Factorization Machine (FM)

**Scripts:**
- Single run: `train_fm.py`
- Sweep: `sweep_fm.py`

## What it is
FM is a lightweight tabular model that adds **pairwise feature interactions** via low-rank factors.

## How to run (100-run sweep)
```bash
python sweep_fm.py --data ann_tensors_full.pt --n-runs 100 --resume
python export_results.py
```

## Key hyperparameters
- `factor_dim`: interaction rank (bigger = more capacity)
- `dropout`, `weight_decay`: regularization

