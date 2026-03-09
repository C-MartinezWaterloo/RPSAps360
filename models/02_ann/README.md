# Model 2 — ANN (categorical embeddings + MLP)

**Scripts:**
- Single run: `train_ann.py`
- Sweeps: `sweep_ann.py`, `sweep_ann_deep_test.py`

## What it is
An MLP on:
- standardized numeric features
- learned embeddings for each categorical column

## How to run (single)
```bash
python train_ann.py --data ann_tensors_full.pt --epochs 30 --batch-size 4096 --lr 0.003 --weight-decay 0.001 \\
  --hidden-dims 2048,1024,512 --dropout 0.0 --embed-dim-cap 64 --seed 42 \\
  --out-csv ann_runs.csv --run-name ann_full_example

python export_results.py
```

## Overfitting signal
The results CSV includes both:
- best-checkpoint metrics (chosen by validation MSE)
- `*_last` metrics (from the final epoch)

