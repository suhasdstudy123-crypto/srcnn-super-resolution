"""Sanity check: does each model actually LEARN, before we spend an hour of GPU on it? (written by us)

Why this exists: our first draft of the deep net trained for 100 epochs on Colab and never
moved off bicubic (val PSNR stuck at 27.377). This script catches that in ~5 minutes on a CPU.

Test (small, but on real photos, with the same degradation and training settings as DIV2K):
    data   : random 96x96 crops; 112->224 style x2 bicubic blur (src/dataset.py degrade)
             train crops from 3 scikit-image photos (astronaut, coffee, rocket),
             val crops from a 4th photo the models never train on (chelsea the cat)
    train  : Adam lr 1e-3, batch 2, MSE loss, a few hundred steps
    report : val PSNR gain over bicubic (dB) at a few steps, and for the deep nets the share
             of "dead" channels before the last layer (ReLU outputs 0 on every val pixel)
A model PASSES if its gain is above 0 dB at the end.

The first-draft deep net (PyTorch default init + zero last layer) is included as a reference.
In a 4-seed test (300 steps) its dead channels were 44-98%; the seed with 98% stayed at
exactly +0.00 dB, like our Colab run. With He init, dead channels were 8-20% for every seed.
Use --seed to try other seeds.

Run:  python improvements/sanity_check.py            (about 5 minutes on a laptop-class CPU)
      python improvements/sanity_check.py --steps 100 --quick   (just the two small models)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import degrade, to_tensor  # noqa: E402
from model import SRCNN  # noqa: E402
from utils import psnr_per_image, set_seed  # noqa: E402

from models import DeepResidualSR, build_model  # noqa: E402

TRAIN_PHOTOS = [data.astronaut(), data.coffee(), data.rocket()]
VAL_PHOTO = data.chelsea()


def random_crops(photos, n, rng, size=96):
    """n random size x size crops -> (blurry, sharp) batches of shape (n, 3, size, size)."""
    blurry, sharp = [], []
    for _ in range(n):
        img = photos[rng.integers(len(photos))]
        top = rng.integers(img.shape[0] - size)
        left = rng.integers(img.shape[1] - size)
        target = Image.fromarray(img[top:top + size, left:left + size])
        blurry.append(to_tensor(degrade(target)))
        sharp.append(to_tensor(target))
    return torch.stack(blurry), torch.stack(sharp)


def first_draft_deep_net():
    """The broken first draft: PyTorch default init + zero-initialised last layer."""
    model = DeepResidualSR()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()                 # back to PyTorch's default init
    nn.init.zeros_(model.body[-1].weight)
    nn.init.zeros_(model.body[-1].bias)
    return model


@torch.no_grad()
def dead_channel_share(model, val_blurry):
    """Fraction of channels before the last layer that output 0 for every val pixel."""
    if not isinstance(model, DeepResidualSR):
        return None
    features = model.body[:-1](val_blurry)                 # everything except the last conv
    return (features.amax(dim=(0, 2, 3)) == 0).float().mean().item()


@torch.no_grad()
def gain_over_bicubic(model, val_blurry, val_sharp):
    pred = model(val_blurry).clamp(0, 1)
    return (psnr_per_image(pred, val_sharp) - psnr_per_image(val_blurry, val_sharp)).mean().item()


def train_and_track(make_model, val, steps, checkpoints, seed):
    """Train a fresh model with MSE; return ({step: val gain over bicubic in dB}, dead share)."""
    set_seed(seed)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)        # same training crops for every model
    gains = {}
    for step in range(1, steps + 1):
        blurry, sharp = random_crops(TRAIN_PHOTOS, 2, rng)
        loss = nn.functional.mse_loss(model(blurry), sharp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            gains[step] = gain_over_bicubic(model, *val)
    return gains, dead_channel_share(model, val[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--n_val", type=int, default=40, help="number of val crops")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="skip the two (slow) deep nets")
    args = p.parse_args()

    val = random_crops([VAL_PHOTO], args.n_val, np.random.default_rng(1))
    checkpoints = sorted({args.steps // 3, 2 * args.steps // 3, args.steps})
    print(f"Bicubic val PSNR on {args.n_val} unseen crops: {psnr_per_image(*val).mean():.2f} dB")
    print("Val PSNR gain over bicubic (dB) at step " + ", ".join(map(str, checkpoints)))

    runs = [("Stanford SRCNN (reference)", SRCNN, None),
            ("imp1 residual SRCNN", lambda: build_model("residual"), True)]
    if not args.quick:
        runs += [("imp2 deep10, He init", lambda: build_model("deep10"), True),
                 ("first draft (default + zero last)", first_draft_deep_net, None)]

    for name, make_model, should_learn in runs:
        t0 = time.time()
        gains, dead = train_and_track(make_model, val, args.steps, checkpoints, args.seed)
        verdict = "" if should_learn is None else ("PASS" if gains[args.steps] > 0 else "FAIL")
        cells = "  ".join(f"{g:+6.2f}" for g in gains.values())
        dead_text = "" if dead is None else f"dead channels {dead * 100:3.0f}%"
        print(f"  {name:34s} {cells}   {verdict:4s} {dead_text:18s} ({time.time() - t0:.0f}s)", flush=True)
    print("PASS = beats bicubic on unseen crops. Many dead channels = the deep net is at risk of getting stuck.")


if __name__ == "__main__":
    main()
