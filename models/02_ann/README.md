# Model 2 — ANN (categorical embeddings + MLP)

**Scripts:**
- Single run: `train_ann.py`
- Sweeps: `sweep_ann.py`, `sweep_ann_deep_test.py`
**Write-up (PDF-style):** `ann_primary_report.tex`

## What it is
An MLP on:
- standardized numeric features
- learned embeddings for each categorical column

## How to run (single)
```bash
python train_ann.py --data ann_tensors_full.pt --epochs 30 --batch-size 512 --lr 0.004 --weight-decay 0.001 \
  --hidden-dims 512,256,128,64 --dropout 0.1 --embed-dim-cap 64 --seed 42 \
  --out-csv ann_runs.csv --run-name full_deep4_bs512_lr4e-3_do0.1_wd1e-3

python export_results.py
```

To evaluate “future prediction” (train on earlier years, test on later years), add:
`--split-strategy time`

## Robustness (multiple seeds + time split)
```bash
python eval_suite.py --data ann_tensors_full.pt --models ann,hedonic \
  --split-strategies random,time --split-seeds 42,7,123 --model-seeds 0,1,42 \
  --out-csv eval_runs.csv

python export_results.py
```

## Overfitting signal
The results CSV includes both:
- best-checkpoint metrics (chosen by validation MSE)
- `*_last` metrics (from the final epoch)
