# Model 4 — DeepFM (FM + MLP)

**Scripts:**
- Single run: `train_deepfm.py`
- Sweep: `sweep_deepfm.py`

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

