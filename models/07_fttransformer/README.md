# Model 7 — FT-Transformer (Transformer for tabular data)

**Script:**
- Single run: `train_fttransformer.py`

## What it is
FT-Transformer is a modern baseline for tabular prediction that represents each feature as a token and models feature interactions with multi-head self-attention.

In this repo (PyTorch-only implementation):
- numeric columns are tokenized into vectors via per-feature affine projections
- categorical columns are embedded into the same `d_model` dimension
- a learnable column embedding is added to each token
- a `[CLS]` token is prepended; the final prediction is produced from the `[CLS]` representation

## How to run (full train; random split)
This reproduces the best full-train FT-Transformer run we logged:
```bash
python train_fttransformer.py --data ann_tensors_full.pt \
  --split-strategy random --seed 42 \
  --d-model 64 --n-heads 4 --n-layers 3 --ff-mult 4 --dropout 0.1 \
  --epochs 6 --batch-size 2048 --lr 0.001 --weight-decay 0.0001 \
  --clip-grad 1.0 --val-max-samples 10000 \
  --run-name ft_full_rand_seed42_e6 --out-csv fttransformer_runs_full_seed42_e6.csv

python export_results.py
```

## Best run so far (full features; full train; seed=42)
From `results_all.csv`:
- Random split: **test RMSE(log1p) ≈ 0.1669**, **test accuracy ≈ 91.42%**
- Time split (new-data proxy): **test RMSE(log1p) ≈ 0.2604**, **test accuracy ≈ 87.71%**

## Practical notes
- On CPU, FT-Transformer is significantly slower than the Embedding+MLP ANN, but can improve time-split generalization.
- If you need faster iterations, set `--train-max-samples 50000` (caps training rows while keeping val/test full).
