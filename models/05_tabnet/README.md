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

## Best run so far (full features, 50k train cap)
From the first 100 sweep runs logged in `results_all.csv`:
- Best `test_acc_pct`: **~78.46%**
- Best `test_rmse` (log1p): **~0.395**

Best config name:
- `tabnet_ed32_cd0.05_d16_a16_s3_g1.3_sh2_in1_do0.1_ls1e-05_e30_bs2048_lr0.003_wd0.0005`

Tip for faster sweeps:
- Use `--val-max-samples 20000` (default) to speed up best-epoch selection while still computing final metrics on full val/test.
