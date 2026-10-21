# Improvements beyond the Stanford project

`src/` is the Stanford replication and is left unchanged. This folder adds three
improvements on top of it, each added to the previous one. Everything else (data,
50/50 val/test split, Adam, lr 0.001, batch 2, 100 epochs, best-on-val, test once) is
identical to the main run, so each difference in the results comes from the improvement alone.

| Run | Model | Loss | What is new |
|---|---|---|---|
| `main` (in `checkpoints/`) | SRCNN 9-1-5, 20,099 weights | MSE | Stanford's method, trained on 800 images |
| `imp1_residual` | Residual SRCNN, 20,099 weights | MSE | predicts only the missing detail: output = input + SRCNN(input) |
| `imp2_deep10` | 10-layer 3x3 residual net, 298,947 weights | MSE | more depth: sees 21x21 pixels instead of 13x13 |
| `imp3_deep10_ssim` | same as imp2 | MSE + 0.01 x (1 - SSIM) | loss that also rewards sharp structure |

The ideas are not ours: residual learning and the deep 3x3 network come from VDSR
(Kim et al., CVPR 2016); the SSIM-based loss follows Zhao et al. (IEEE TCI 2017).
Our part is applying them to the Stanford setup and measuring each one under identical conditions.

## Files
- `models.py`: `ResidualSRCNN` (improvement 1) and `DeepSR` (improvement 2)
- `losses.py`: MSE and MSE + SSIM (improvement 3), with a differentiable SSIM
- `train_improved.py`: training script; reuses the dataset, training loop and metrics from `src/`
- `demo_improved.py`: Bicubic | Stanford SRCNN | Improved | Original, side by side

## Run (Colab, after the setup cells)
```
python improvements/train_improved.py --cache_dir <cache> --out_dir <drive>/checkpoints --run_name imp1_residual --arch srcnn_res
python improvements/train_improved.py --cache_dir <cache> --out_dir <drive>/checkpoints --run_name imp2_deep10 --arch deep --n_layers 10
python improvements/train_improved.py --cache_dir <cache> --out_dir <drive>/checkpoints --run_name imp3_deep10_ssim --arch deep --n_layers 10 --loss mse_ssim
```
