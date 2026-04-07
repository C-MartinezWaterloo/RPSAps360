# Model 8 — TabTransformer (attention over categorical embeddings)

**Script:**
- Single run: `train_tabtransformer.py`

## What it is
TabTransformer applies Transformer attention to **categorical feature embeddings** to learn context-dependent representations (e.g., how a neighborhood category interacts with time and property style). The transformed categorical representation is then combined with numeric features and passed through an MLP head for regression.

In this repo (PyTorch-only implementation):
- each categorical column is embedded into `d_model`
- a Transformer encoder mixes information across categorical columns
- numeric features are concatenated after a learned projection
- an MLP head predicts `log1p(TransactionPrice)`

## How to run (full train; random split)
```bash
python train_tabtransformer.py --data ann_tensors_full.pt \
  --split-strategy random --seed 42 \
  --d-model 64 --n-heads 4 --n-layers 3 --ff-mult 4 --dropout 0.1 \
  --mlp-hidden-dims 256,128 \
  --epochs 6 --batch-size 2048 --lr 0.001 --weight-decay 0.0001 \
  --clip-grad 1.0 \
  --run-name tabtransformer_full_rand_seed42_e6 --out-csv tabtransformer_runs_full_seed42_e6.csv

python export_results.py
```

## Best run so far (full features; full train; seed=42)
From `results_all.csv`:
- Random split: **test RMSE(log1p) ≈ 0.1997**, **test accuracy ≈ 88.97%**
- Time split (new-data proxy): **test RMSE(log1p) ≈ 0.2730**, **test accuracy ≈ 79.06%**

## Practical notes
- Compared to the Embedding+MLP ANN, TabTransformer is slower on CPU and (in our runs) did not outperform on either split.
- It is still a useful attention-based baseline to show that “adding attention” does not automatically improve tabular performance without careful tuning.
