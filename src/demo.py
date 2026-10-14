"""Live demo: load the trained model (no training) and show bicubic vs SRCNN vs original.

Two ways to use it:
    # a test image (index 0-49), the honest choice: the model never saw these
    python src/demo.py --ckpt <drive>/checkpoints/main/best.pth --cache_dir <cache> --index 7
    # any photo of your own (center-cropped and resized to 224 like the dataset)
    python src/demo.py --ckpt <drive>/checkpoints/main/best.pth --image my_photo.jpg

Saves a figure: top row = full images with PSNR, bottom row = zoom on the region
where SRCNN helped most, so the difference is visible to the audience.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import degrade, load_target, split_val_test, to_tensor
from model import SRCNN
from utils import psnr_per_image, ssim_per_image


def get_target(args):
    """The sharp 224x224 target, from the test set or from a user photo."""
    if args.image:
        return load_target(args.image)
    valid = np.load(Path(args.cache_dir) / "valid_224.npy")
    _, test_idx = split_val_test(len(valid))
    return Image.fromarray(valid[test_idx[args.index]])


def best_zoom_box(blurry, pred, sharp, size=56):
    """Find the size x size window where SRCNN reduced the error the most."""
    gain = ((blurry - sharp) ** 2 - (pred - sharp) ** 2).mean(dim=1, keepdim=True)  # (1,1,H,W)
    pooled = F.avg_pool2d(gain, size, stride=8)
    row, col = divmod(pooled.flatten().argmax().item(), pooled.shape[-1])
    return row * 8, col * 8


def to_img(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--index", type=int, default=0, help="test image number, 0-49")
    p.add_argument("--image", default=None, help="path to your own photo instead of a test image")
    p.add_argument("--out", default="results/demo.png")
    args = p.parse_args()

    model = SRCNN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    target = get_target(args)
    sharp = to_tensor(target).unsqueeze(0)            # add a batch dimension: (1, 3, 224, 224)
    blurry = to_tensor(degrade(target)).unsqueeze(0)  # 224 -> 112 -> 224, bicubic
    with torch.no_grad():
        pred = model(blurry).clamp(0, 1)

    scores = {name: (psnr_per_image(x, sharp).item(), ssim_per_image(x, sharp).item())
              for name, x in [("Bicubic", blurry), ("SRCNN", pred)]}
    for name, (ps, ss) in scores.items():
        print(f"{name:8s} PSNR {ps:.2f} dB | SSIM {ss:.4f}")

    top, left = best_zoom_box(blurry, pred, sharp)
    panels = [("Bicubic (input)", blurry), ("SRCNN (ours)", pred), ("Original", sharp)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for col, (title, x) in enumerate(panels):
        img = to_img(x)
        label = title if title == "Original" else f"{title}\n{scores[title.split()[0]][0]:.2f} dB"
        axes[0, col].imshow(img)
        axes[0, col].add_patch(plt.Rectangle((left, top), 56, 56, fill=False, ec="#eda100", lw=2))
        axes[0, col].set_title(label, fontsize=11)
        axes[1, col].imshow(img[top:top + 56, left:left + 56], interpolation="nearest")
        axes[1, col].set_title("zoom", fontsize=9, color="#52514e")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
