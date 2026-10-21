"""Side-by-side demo: Bicubic | Stanford SRCNN | Improved model | Original.

    python improvements/demo_improved.py --improved <ckpts>/imp3_deep10_ssim/best.pth \
        --stanford checkpoints/main/best.pth --cache_dir <cache> --index 7
    (or --image my_photo.jpg instead of --cache_dir/--index)

Each model is rebuilt from the config.json saved next to its best.pth.
Reuses the image loading and zoom helpers from src/demo.py.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import degrade, to_tensor  # noqa: E402
from demo import best_zoom_box, get_target, to_img  # noqa: E402
from utils import psnr_per_image, ssim_per_image  # noqa: E402

from models import build_model  # noqa: E402


def load(ckpt):
    """Rebuild the model from its saved settings and load the weights."""
    config_file = Path(ckpt).parent / "config.json"
    settings = json.loads(config_file.read_text()) if config_file.exists() else {}
    settings.setdefault("arch", "srcnn")              # Stanford-replication runs have no 'arch'
    model = build_model(**settings)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--improved", required=True)
    p.add_argument("--stanford", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--image", default=None)
    p.add_argument("--out", default="improvements/results/demo_improved.png")
    args = p.parse_args()

    target = get_target(args)
    sharp = to_tensor(target).unsqueeze(0)
    blurry = to_tensor(degrade(target)).unsqueeze(0)
    with torch.no_grad():
        stanford = load(args.stanford)(blurry).clamp(0, 1)
        improved = load(args.improved)(blurry).clamp(0, 1)

    panels = [("Bicubic", blurry), ("Stanford SRCNN", stanford), ("Improved (ours)", improved), ("Original", sharp)]
    for name, x in panels[:3]:
        print(f"{name:16s} PSNR {psnr_per_image(x, sharp).item():.2f} dB | SSIM {ssim_per_image(x, sharp).item():.4f}")

    top, left = best_zoom_box(blurry, improved, sharp)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for col, (name, x) in enumerate(panels):
        img = to_img(x)
        title = name if name == "Original" else f"{name}\n{psnr_per_image(x, sharp).item():.2f} dB"
        axes[0, col].imshow(img)
        axes[0, col].add_patch(plt.Rectangle((left, top), 56, 56, fill=False, ec="#eda100", lw=2))
        axes[0, col].set_title(title, fontsize=11)
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
