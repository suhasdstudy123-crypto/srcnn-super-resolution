"""Improved super-resolution models (written by us).

Two ideas, added one at a time on top of the Stanford SRCNN:

imp1  ResidualSRCNN  - the SAME Stanford SRCNN (imported unchanged from src/model.py),
                       but it predicts only the missing detail (the "residual"):
                           output = input + SRCNN(input)
                       The blurry input is already close to the answer, so the network
                       only has to learn the small difference (edges, texture).

imp2  DeepResidualSR - VDSR-style network (Kim et al., CVPR 2016), shortened to 10 layers:
                           conv 3x3 (3->64) + ReLU
                           8 x [conv 3x3 (64->64) + ReLU]
                           conv 3x3 (64->3)
                           output = input + network(input)
                       Many small 3x3 layers see a 21x21 neighbourhood in total (each layer
                       adds 2 pixels), with ~15x the parameters of SRCNN.

Initialisation matters for the deep net. PyTorch's default init makes the signal shrink at
every layer; after 10 layers it is almost zero (mean |activation| ~0.01). Our first draft
combined that with a zero-initialised last layer, and it stayed stuck at exactly bicubic.
He/Kaiming init (He et al., 2015; used by VDSR) picks the starting weight size so the signal
keeps roughly the same strength through every ReLU layer. Run `python improvements/models.py`
to see the difference layer by layer, and improvements/sanity_check.py for the learning test.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # so we can import src/ files
from model import SRCNN, count_params  # noqa: E402  (Stanford model, unchanged)


class ResidualSRCNN(nn.Module):
    """imp1: Stanford SRCNN that predicts the residual (missing detail) instead of the full image."""

    def __init__(self):
        super().__init__()
        self.body = SRCNN()          # exactly the replication model, same default init

    def forward(self, x):
        return x + self.body(x)      # blurry input + predicted detail


class DeepResidualSR(nn.Module):
    """imp2: deep stack of 3x3 conv layers (VDSR-style) with a global residual connection."""

    def __init__(self, depth=10, channels=64, num_channels=3):
        super().__init__()
        layers = [nn.Conv2d(num_channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(channels, num_channels, 3, padding=1))  # last layer: no ReLU
        self.body = nn.Sequential(*layers)
        init_he(self)

    def forward(self, x):
        return x + self.body(x)


def init_he(model):
    """He/Kaiming init: weight std = sqrt(2 / fan_in) for layers followed by ReLU.

    fan_in = inputs to one output pixel = in_channels * 3 * 3. The factor 2 makes up for
    ReLU zeroing about half the values. The LAST conv has no ReLU after it, so it uses
    gain 1 (std = sqrt(1 / fan_in)). Biases start at 0.
    """
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    for i, conv in enumerate(convs):
        last = i == len(convs) - 1
        nn.init.kaiming_normal_(conv.weight, nonlinearity="linear" if last else "relu")
        nn.init.zeros_(conv.bias)


MODELS = {"residual": ResidualSRCNN, "deep10": DeepResidualSR}


def build_model(name):
    """Create a model by name: 'residual' (imp1) or 'deep10' (imp2 and imp3)."""
    return MODELS[name]()


@torch.no_grad()
def activation_sizes(model, x):
    """Mean |activation| after every ReLU of a DeepResidualSR, to check the signal survives."""
    sizes = []
    for layer in model.body:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            sizes.append(x.abs().mean().item())
    return sizes


if __name__ == "__main__":
    # Quick self-test: python improvements/models.py
    torch.manual_seed(0)
    x = torch.rand(2, 3, 224, 224)
    for name in MODELS:
        m = build_model(name)
        print(f"{name:8s} params {count_params(m):7,d} | output shape {tuple(m(x).shape)}")

    he = DeepResidualSR()
    default = DeepResidualSR()
    for conv in default.modules():          # undo our init -> PyTorch's default init
        if isinstance(conv, nn.Conv2d):
            conv.reset_parameters()
    print("\nMean |activation| after each ReLU (input mean is ~0.5):")
    print("  He init     :", " ".join(f"{v:.3f}" for v in activation_sizes(he, x)))
    print("  default init:", " ".join(f"{v:.3f}" for v in activation_sizes(default, x)))
