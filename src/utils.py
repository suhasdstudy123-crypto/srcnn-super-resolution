"""Small helpers: PSNR metric and seeding.

PSNR formula adapted from yjn870/SRCNN-pytorch (utils.py: calc_psnr).
Change: we compute PSNR per image and return all of them, so we can report the mean
of per-image PSNR (like Stanford) and also study which images are hard.
"""
import random

import numpy as np
import torch


def psnr_per_image(pred, target, eps=1e-10):
    """PSNR in dB for each image in a batch.

    pred, target: tensors of shape (N, C, H, W) with pixel values in [0, 1].
    Returns a tensor of N values.
    """
    pred = pred.clamp(0, 1)                                # model output can go slightly outside [0,1]
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))       # one MSE per image
    return 10 * torch.log10(1.0 / (mse + eps))             # max pixel value is 1, so MAX^2 = 1


def set_seed(seed=42):
    """Make runs repeatable: same split, same initial weights, same shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Quick self-test: python src/utils.py
    target = torch.rand(2, 3, 224, 224)
    print("Identical images (should be ~100 dB, capped by eps):", psnr_per_image(target, target))
    noisy = target + 0.1 * torch.randn_like(target)
    print("Noise std 0.1 (should be roughly 20 dB):", psnr_per_image(noisy, target))
