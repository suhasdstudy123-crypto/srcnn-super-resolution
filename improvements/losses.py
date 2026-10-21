"""Loss functions. 'mse' is Stanford's; 'mse_ssim' is improvement 3.

mse_ssim = MSE + ssim_weight * (1 - SSIM)
  - MSE keeps colours and brightness accurate.
  - (1 - SSIM) is 0 for a perfect image; lowering it keeps edges and textures sharp.
  - ssim_weight = 0.01 makes both parts about the same size (MSE ~0.002, 1-SSIM ~0.1).
Follows the idea of Zhao et al. 2017, "Loss Functions for Image Restoration with Neural Networks".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_torch(pred, target, window=11, sigma=1.5):
    """Mean SSIM of a batch, written in PyTorch so gradients flow through it.

    Same formula as scikit-image, with an 11x11 Gaussian window. Used only for training;
    the SSIM numbers we report still come from scikit-image (src/utils.py).
    """
    c1, c2 = 0.01 ** 2, 0.03 ** 2                         # small constants that avoid dividing by 0
    coords = torch.arange(window, dtype=pred.dtype, device=pred.device) - window // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g).expand(3, 1, window, window)  # one window per colour channel

    def local_mean(x):
        return F.conv2d(x, kernel, groups=3)

    mu_p, mu_t = local_mean(pred), local_mean(target)
    var_p = local_mean(pred * pred) - mu_p ** 2              # brightness variation (contrast)
    var_t = local_mean(target * target) - mu_t ** 2
    cov = local_mean(pred * target) - mu_p * mu_t            # do the patterns line up? (structure)
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))
    return ssim_map.mean()


def make_loss(name="mse", ssim_weight=0.01):
    mse = nn.MSELoss()
    if name == "mse":
        return mse
    if name == "mse_ssim":
        return lambda pred, target: mse(pred, target) + ssim_weight * (1 - ssim_torch(pred, target))
    raise ValueError(f"unknown loss {name}")


if __name__ == "__main__":
    # Quick self-test: python improvements/losses.py
    t = F.avg_pool2d(torch.rand(2, 3, 64, 64), 3, 1, 1)
    noisy = (t + 0.05 * torch.randn_like(t)).clamp(0, 1).requires_grad_(True)
    print("SSIM identical:", round(ssim_torch(t, t).item(), 4), "| SSIM noisy:", round(ssim_torch(noisy, t).item(), 4))
    loss = make_loss("mse_ssim")(noisy, t)
    loss.backward()
    print("mse_ssim loss:", round(loss.item(), 5), "| gradients flow:", bool(noisy.grad.abs().sum() > 0))
