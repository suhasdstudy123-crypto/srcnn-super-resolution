# Improvements beyond the Stanford SRCNN

`src/` is the replication of the Stanford CS229 project and is left unchanged. This folder
adds three improvements, each built on top of the previous one. Everything except the
model and the loss is identical to the replication run `main`: the same 800 DIV2K training
images, the same 50 val / 50 test split (seed 42), the same 224x224 setup (x2 bicubic
degradation), Adam with lr 1e-3, batch size 2, 100 epochs, best-on-val weights, and one
test pass at the end. Any difference in the results therefore comes from the improvement.

The ideas are not ours. Residual learning and the deep 3x3 network come from VDSR
(Kim et al., CVPR 2016), and the SSIM-based loss follows Zhao et al. (IEEE TCI 2017).
Our part is applying them to the Stanford setup and measuring each one under the same
conditions.

## Results (50 test images, never used for training or model selection)

| Method | Test PSNR (dB) | Test SSIM | Gain over bicubic | Gain over Stanford | Params | s/epoch (T4) |
|---|---|---|---|---|---|---|
| Bicubic | 26.41 | 0.864 | - | - | - | - |
| Stanford SRCNN (`main`) | 28.03 | 0.900 | +1.62 dB | - | 20,099 | ~8 |
| imp1: residual SRCNN | 28.27 | 0.904 | +1.86 dB | +0.24 dB | 20,099 | 7.8 |
| imp2: deep10 residual net | **29.05** | 0.916 | **+2.64 dB** | **+1.02 dB** | 298,947 | 39.0 |
| imp3: deep10 + SSIM loss | 28.98 | **0.920** | +2.58 dB | +0.96 dB | 298,947 | 41.5 |

- imp2 beats the Stanford SRCNN on **all 50 of the 50** test images (`results/per_image_gain.png`).
- imp3 gives up 0.06 dB of PSNR for the best SSIM. This is the expected tradeoff of an SSIM
  loss, and the weight sweep below shows the same pattern.
- Stanford's validation PSNR drops by 1-3 dB at several epochs during training. All three
  residual models train smoothly (`results/val_curves.png`).
- Cost: imp2 and imp3 have about 15x the weights and take about 5x longer per epoch than
  SRCNN. imp1 improves on SRCNN with no extra weights.

Source of every number: `results/improvement_table.csv` (made by `compare.py`). Before
plotting, `compare.py` reloads each `best.pth`, re-scores it on the test images and checks
that it reproduces the run's `test.csv`.

## The three improvements

**imp1, residual SRCNN** (`models.py: ResidualSRCNN`). This is the Stanford SRCNN unchanged,
but it predicts only the missing detail: `output = input + SRCNN(input)`. The blurry input
is already close to the target, so the network learns a small correction and does not have
to copy the whole image through three layers. After epoch 1 its val PSNR was 28.31 dB,
compared with 24.15 dB for `main`.

**imp2, deep residual net** (`models.py: DeepResidualSR`). It has ten 3x3 conv layers with
64 channels and ReLU, keeps the same `input +` skip, and has no ReLU after the last layer
because the residual can be negative. Ten 3x3 layers together see a 21x21 pixel area, while
SRCNN 9-1-5 sees 13x13. The weights use He/Kaiming initialisation, as in VDSR.

**imp3, MSE + SSIM loss** (`losses.py`). The loss is `MSE + 0.2 x (1 - SSIM)`. MSE keeps
colours and brightness right. The SSIM term rewards correct local structure (edges and
texture). Our differentiable SSIM uses exactly the scikit-image settings that we use to
report SSIM (7x7 window, sample covariance), and it matches scikit-image to 4 decimals.

### How the SSIM weight was chosen (val only)
We trained the deep10 net for 10 epochs with weights 0, 0.01, 0.05 and 0.2, and averaged
epochs 8-10 on the val set. The selection rule was fixed before the runs: highest val SSIM
with val PSNR within 0.1 dB of weight 0, and an SSIM gain above 0.002. The rule picked
**0.2**, with PSNR -0.098 dB and SSIM +0.0039 relative to weight 0. It was close to the
limit: a slightly stricter PSNR budget would have picked 0.05. See
`results/ssim_weight_sweep.csv` and `results/ssim_sweep.png`.

## A lesson: why our first deep net never learned
Our first draft of the deep net used PyTorch's default initialisation and a zero-initialised
last layer. On Colab it stayed at exactly bicubic PSNR (27.377) for all 100 epochs.
`sanity_check.py` reproduces this on a CPU in a few minutes:

- With default init the signal shrinks layer by layer: mean |activation| before the last
  layer is about 0.01, compared with about 0.25 with He init (`python improvements/models.py`).
- Most ReLUs then go "dead" (they output 0 on every pixel and get no gradient). In a 4-seed
  test, 44-98% of the channels before the last layer were dead. The seed with 98% dead
  channels stayed at exactly +0.00 dB over bicubic, as on Colab.
- With He init, only 8-20% of channels were dead for every seed, and the model learned each time.

The stall depends on the seed (1 of 4 seeds in our test). He init removes the risk; it does
not guarantee that every other setting will train well.

## Files
| File | What it does |
|---|---|
| `models.py` | `ResidualSRCNN` (imp1), `DeepResidualSR` with He init (imp2/imp3), `build_model` |
| `losses.py` | Differentiable SSIM (scikit-image settings) and `make_loss("mse" / "mse_ssim")` |
| `train_improved.py` | Training script; reuses `get_datasets`, `train_one_epoch`, `evaluate` from `src/` |
| `sanity_check.py` | A 300-step CPU test that each model beats bicubic, plus the dead-ReLU check |
| `compare.py` | Results table and all the figures in `results/` |
| `results/` | `improvement_table.csv`, `per_image_psnr.csv`, `ssim_weight_sweep.csv`, figures, demos |
| `checkpoints/<run>/` | `best.pth`, `config.json`, `history.csv`, `test.csv`, `test_per_image.csv` |

## Reproduce (Google Colab, T4 GPU)
Data setup is the same as for the replication (the DIV2K cache `train_224.npy` / `valid_224.npy`).

```
# optional CPU check (about 3 minutes)
python improvements/sanity_check.py

# the three runs (about 15, 65 and 70 minutes on a T4)
python improvements/train_improved.py --cache_dir <cache> --out_dir <out> --run_name imp1_residual --model residual
python improvements/train_improved.py --cache_dir <cache> --out_dir <out> --run_name imp2_deep10 --model deep10
python improvements/train_improved.py --cache_dir <cache> --out_dir <out> --run_name imp3_deep10_ssim \
    --model deep10 --loss mse_ssim --ssim_weight 0.2

# SSIM-weight sweep (val only, the test set is not touched)
python improvements/train_improved.py --cache_dir <cache> --out_dir <out>/sweep --run_name sweep_w0.05 \
    --model deep10 --loss mse_ssim --ssim_weight 0.05 --epochs 10 --no_test

# table and figures (works on CPU)
python improvements/compare.py --stanford_dir checkpoints --imp_dir <out> --baselines results/baselines.csv \
    --cache_dir <cache> --sweep_csv improvements/results/ssim_weight_sweep.csv --out_dir improvements/results
```
If Colab disconnects, run the same command again: training resumes from `last.pth`, which is
kept on Drive and not committed.

Notes: imp1 was resumed once after a Colab disconnect. The weights and optimizer state were
restored exactly; only the shuffle order of the training images restarts from the seed.
Results come from one run per model (seed 42), so differences of a few hundredths of a dB
(such as imp2 vs imp3 PSNR) are within run-to-run noise.

## References
- Dong et al., "Image Super-Resolution Using Deep Convolutional Networks", TPAMI 2016 (SRCNN).
- Kim, Lee, Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks", CVPR 2016 (VDSR).
- Zhao, Gallo, Frosio, Kautz, "Loss Functions for Image Restoration with Neural Networks", IEEE TCI 2017.
- He et al., "Delving Deep into Rectifiers", ICCV 2015 (He/Kaiming initialisation).
- Garber, Grossman, Johnson-Yu, "Image Super-Resolution Via a Convolutional Neural Network", Stanford CS229, 2020.
