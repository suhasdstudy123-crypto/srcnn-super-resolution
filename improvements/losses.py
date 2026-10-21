"""Loss functions (written by us). 'mse' is Stanford's loss; 'mse_ssim' is improvement 3.

    mse_ssim = MSE + ssim_weight * (1 - SSIM)

- MSE punishes every pixel error equally, so it keeps colours and brightness right, but it
  is happy with slightly blurry edges (blur gives a small average error).
- SSIM compares local brightness, contrast and structure (are edges in the same places?).
  1 - SSIM is 0 for a perfect image, so lowering it pushes the model towards sharp structure.
- ssim_weight balances the two. Idea from Zhao et al. 2017, "Loss Functions for Image
  Restoration with Neural Networks" (they mix L1 and MS-SSIM; we keep MSE, Stanford's loss).

Our SSIM uses exactly the settings of scikit-image's structural_similarity, which we use to
REPORT SSIM (src/utils.py): 7x7 uniform window, data range 1, K1 0.01, K2 0.03, sample
covariance, border of 3 pixels ignored. So the model is trained on the same SSIM we report.
The only difference is that this one is written in PyTorch, so gradients can flow through it.
"""
import torch.nn as nn
import torch.nn.functional as F

WIN = 7                       # scikit-image default window size
C1, C2 = 0.01 ** 2, 0.03 ** 2  # (K1 * data_range)^2, (K2 * data_range)^2: avoid dividing by 0


def ssim_torch(pred, target):
    """Mean SSIM over a batch (N, C, H, W) in [0, 1], averaged over channels like scikit-image."""
    mean = lambda x: F.avg_pool2d(x, WIN, stride=1)        # local 7x7 average, no padding
    cov_norm = WIN * WIN / (WIN * WIN - 1)                  # sample covariance (divide by n-1)

    mu_p, mu_t = mean(pred), mean(target)                   # local brightness
    var_p = cov_norm * (mean(pred * pred) - mu_p ** 2)      # local contrast
    var_t = cov_norm * (mean(target * target) - mu_t ** 2)
    cov = cov_norm * (mean(pred * target) - mu_p * mu_t)    # do the patterns line up? (structure)

    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * cov + C2)) / (
        (mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))
    return ssim_map.mean()     # no padding above, so the 3-pixel border is already left out


def make_loss(name="mse", ssim_weight=0.01):
    """Return a loss function loss(pred, target) by name."""
    mse = nn.MSELoss()
    if name == "mse":
        return mse
    if name == "mse_ssim":
        return lambda pred, target: mse(pred, target) + ssim_weight * (1 - ssim_torch(pred, target))
    raise ValueError(f"unknown loss {name}")


if __name__ == "__main__":
    # Quick self-test: python improvements/losses.py
    import sys
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image
    from skimage import data

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from dataset import degrade, to_tensor
    from utils import ssim_per_image

    target = Image.fromarray(data.astronaut()).resize((224, 224), Image.BICUBIC)
    sharp = to_tensor(target).unsqueeze(0)
    blurry = to_tensor(degrade(target)).unsqueeze(0)
    noisy = (sharp + 0.05 * torch.randn_like(sharp)).clamp(0, 1)
    for name, x in [("identical", sharp), ("bicubic", blurry), ("noisy", noisy)]:
        print(f"SSIM {name:9s}: ours {ssim_torch(x, sharp).item():.4f} | "
              f"scikit-image {ssim_per_image(x, sharp).item():.4f}")

    pred = blurry.clone().requires_grad_(True)
    loss = make_loss("mse_ssim")(pred, sharp)
    loss.backward()
    print(f"mse_ssim loss on bicubic: {loss.item():.5f} | gradients flow: {bool(pred.grad.abs().sum() > 0)}")
