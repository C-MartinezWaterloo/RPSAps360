# Model 5 — TabNet (sequential feature selection)

**Scripts:**
- Single run: `train_tabnet.py`
- Sweep: `sweep_tabnet.py`

## What it is
TabNet is a tabular neural network that performs **sequential feature selection** via sparse attention masks (sparsemax).

In this repo:
- categoricals are embedded and concatenated with numeric features
- TabNet selects and transforms features over multiple decision steps

## How to run (lots of runs)
```bash
python sweep_tabnet.py --data ann_tensors_full.pt --n-runs 200 --resume
python export_results.py
```

