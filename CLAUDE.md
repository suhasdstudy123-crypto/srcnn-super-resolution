# CLAUDE.md — SRCNN Super-Resolution (ML Mini Project)

## What this project is
Course mini project (UE24CS352A, team of 2). Replicate Stanford CS229 project #4,
"Image Super-Resolution Via a Convolutional Neural Network" (Garber, Grossman, Johnson-Yu, 2020).
The authors published NO code. Our model and PSNR code are adapted from
https://github.com/yjn870/SRCNN-pytorch (MIT); everything else is written by us.
The README must say this clearly.

## The user
Suhas, B.Tech CSE (AI/ML). Knows ML basics. Wants:
- Small steps, one file at a time. Wait for confirmation before the next step.
- After writing each file, a simple explanation he can defend in viva/Q&A.
- Some functions kept short enough that he could rewrite them himself.
- Direct, honest tradeoffs. No HTML tags (like <sub>) in explanations.

## Setup to replicate (Stanford)
- Model: SRCNN, 3 conv layers, RGB in/out:
  conv 9x9, 3->64, ReLU -> conv 1x1, 64->32, ReLU -> conv 5x5, 32->3. "Same" padding (4, 0, 2).
  About 20k parameters. Output size = input size = 224x224.
- Data: DIV2K_valid_HR (100 images), split 60/20/20 with a fixed seed.
  Target: center crop 800x800 -> resize to 224x224.
  Input: target -> downscale to 112x112 -> upscale back to 224x224.
  Interpolation: BICUBIC by default (Stanford's report says bicubic, poster says bilinear;
  keep it a config option and state the choice in the write-up).
- Training: MSE loss, Adam, lr 1e-3, batch size 2, 100 epochs. Metric: PSNR (pixels in [0,1]).
- Stanford results (test PSNR, dB): nearest 20.551, bilinear 23.182, bicubic 23.226, SRCNN 26.442.
  Max val 27.455, max train 28.03.
- Ablations (all worse or equal): weight decay 26.429, tanh output 26.282, BatchNorm 24.373.
  LR sweep {1e-2, 1e-3, 1e-4}: 1e-3 best.

## Extensions (after replication works)
Train on DIV2K_train_HR (800 images) with larger batch; add SSIM; try to beat 26.44 dB.

## Repo layout
```
src/            dataset.py, model.py, train.py, baselines.py, utils.py
notebooks/      run_experiments.ipynb  (THIN: only setup + calls into src/, no logic)
results/        small CSVs and PNG plots (committed)
CLAUDE.md, README.md, .gitignore
```
Never commit: data/, checkpoints/, *.pth, *.zip. Data and checkpoints live on Google Drive.

## Workflow
- Code is written/edited on the laptop (Windows, D:\ML-MiniProject). Use Windows-friendly
  paths (pathlib) and commands. Code must also run unchanged on Colab (Linux, CUDA).
- GitHub is the bridge. Colab only does `git pull` and runs the notebook.
- Every script takes CLI args (argparse) for paths, epochs, batch size, lr, n_images, device,
  so the same code does a tiny CPU smoke test locally and a full GPU run on Colab.
- Before pushing: smoke test on CPU (e.g. 4 images, 1 epoch).
- On Colab: copy images to /content for speed; save checkpoints + results to Drive after
  every run (Colab disconnects). Drive folder:
  "Image Super-Resolution Via a Convolutional Neural Network-ML".
- Set seeds (random, numpy, torch) everywhere for reproducibility.
- Small commits with clear messages (e.g. "Add DIV2K dataset class with 60/20/20 split").
  Repo maintenance is graded.

## Conventions
- Python 3.10+, PyTorch, torchvision, Pillow, numpy, matplotlib, scikit-image (for SSIM).
- PSNR = 10 * log10(1 / MSE) for images in [0,1]; report the mean of per-image PSNR.
- Keep results tables as CSV in results/ so plots and the write-up read from one source.
