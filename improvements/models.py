"""Improved models (our extensions beyond the Stanford project).

Improvement 1  ResidualSRCNN: Stanford's exact 3-layer SRCNN, but it predicts only the
               missing detail:  output = blurry input + SRCNN(blurry input)
Improvement 2  DeepSR: 10 layers of 3x3 convs (VDSR-style, Kim et al. 2016), also residual.
Improvement 3  uses DeepSR with a different loss (see losses.py), not a new model.

Both residual models start with their last layer set to zero, so before any training
they output exactly the bicubic input, and training only learns the improvement on top.
"""
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # reuse the original code
from model import SRCNN  # noqa: E402  (Stanford's model, unchanged)


def zero_last_layer(conv):
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)


class ResidualSRCNN(SRCNN):
    """Same layers and the same 20,099 weights as SRCNN; only the output equation changes."""

    def __init__(self):
        super().__init__()
        zero_last_layer(self.conv3)

    def forward(self, x):
        return x + super().forward(x)   # input + predicted details


class DeepSR(nn.Module):
    """n_layers conv layers, 3x3, 64 feature maps, ReLU between them, residual output.

    Each 3x3 layer widens the area a pixel 'sees' by 2, so 10 layers see 21x21
    (Stanford's 9-1-5 SRCNN sees 13x13).
    """

    def __init__(self, n_layers=10, channels=64):
        super().__init__()
        layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        last = nn.Conv2d(channels, 3, 3, padding=1)
        zero_last_layer(last)
        self.body = nn.Sequential(*layers, last)

    def forward(self, x):
        return x + self.body(x)


def build_model(arch="srcnn_res", n_layers=10, **_):
    """Create the model named in the run settings (used by training and the demo)."""
    if arch == "srcnn_res":
        return ResidualSRCNN()
    if arch == "deep":
        return DeepSR(n_layers=n_layers)
    if arch == "srcnn":                       # lets the demo load Stanford's original run too
        return SRCNN()
    raise ValueError(f"unknown arch {arch}")


if __name__ == "__main__":
    # Quick self-test: python improvements/models.py
    import torch
    from model import count_params

    x = torch.rand(2, 3, 224, 224)
    for name in ["srcnn_res", "deep"]:
        m = build_model(name)
        print(f"{name:10s} output {tuple(m(x).shape)} | parameters {count_params(m):,} "
              f"| starts as the input exactly: {torch.equal(m(x), x)}")
