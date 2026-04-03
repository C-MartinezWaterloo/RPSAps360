# Model 4 — DeepFM (FM + MLP)

**Scripts:**
- Single run: `train_deepfm.py`
- Sweep: `sweep_deepfm.py`
**Write-up (PDF-style):** `deepfm_report.tex`

## What it is
DeepFM combines:
- a linear (“wide”) part
- FM pairwise interactions (low-rank)
- a deep MLP over embeddings + numeric features

## How to run (lots of runs)
```bash
python sweep_deepfm.py --data ann_tensors_full.pt --n-runs 200 --resume
python export_results.py
```

## Best config we found (so far)
From the 200-run sweep on `ann_tensors_full.pt` with `train_max_samples=50,000`:
- `factor_dim=16`
- `hidden_dims=2048,1024,512`
- `dropout=0.05`
- `epochs=40`, `batch_size=4096`, `lr=0.002`, `weight_decay=0.001`
- `test_acc_pct≈89.70%`, `test_rmse≈0.2851`
