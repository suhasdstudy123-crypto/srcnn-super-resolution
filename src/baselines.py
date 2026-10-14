"""Interpolation baselines: how good is plain upscaling, with no learning at all?

For each test (and val) image:
    input  = target shrunk to 112x112 and enlarged back to 224x224 with method M
    score  = PSNR / SSIM / MSE of that input against the sharp target
for M in {nearest, bilinear, bicubic}. SRCNN must beat bicubic to be worth anything.
Stanford's test PSNR: nearest 20.551, bilinear 23.182, bicubic 23.226.

Run:  python src/baselines.py --cache_dir <dir with valid_224.npy> --out results/baselines.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataset import degrade, split_val_test, to_tensor
from utils import psnr_per_image, ssim_per_image

METHODS = ["nearest", "bilinear", "bicubic"]


def evaluate_method(targets, method):
    """Mean PSNR, SSIM and MSE of `method` upscaling over a set of uint8 target images."""
    sharp = torch.stack([to_tensor(Image.fromarray(t)) for t in targets])
    blurry = torch.stack([to_tensor(degrade(Image.fromarray(t), interp=method)) for t in targets])
    return {
        "psnr": psnr_per_image(blurry, sharp).mean().item(),
        "ssim": ssim_per_image(blurry, sharp).mean().item(),
        "mse": ((blurry - sharp) ** 2).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out", default="results/baselines.csv")
    args = parser.parse_args()

    valid = np.load(Path(args.cache_dir) / "valid_224.npy")
    val_idx, test_idx = split_val_test(len(valid))

    rows = []
    for split, idx in [("val", val_idx), ("test", test_idx)]:
        for method in METHODS:
            rows.append({"split": split, "method": method, **evaluate_method(valid[idx], method)})

    df = pd.DataFrame(rows).round(4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
