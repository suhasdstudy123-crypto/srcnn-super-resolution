"""SRCNN model (Dong et al., 2014) in the RGB 9-1-5 setup used by the Stanford CS229 project.

Adapted from yjn870/SRCNN-pytorch (models.py). Changes from that repo:
  - 3 input/output channels (RGB) instead of 1 (Y channel only)
  - middle layer is 1x1 (Stanford) instead of 5x5
  - optional BatchNorm and tanh output, used only for the ablation experiments
"""
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=3, use_bn=False, tanh_out=False):
        super().__init__()
        # Layer 1: patch extraction. 9x9 filters, 64 feature maps. padding=4 keeps size the same.
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        # Layer 2: non-linear mapping. 1x1 filters, 64 -> 32 feature maps. No padding needed.
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        # Layer 3: reconstruction. 5x5 filters, back to 3 channels (RGB). padding=2.
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)
        # Ablation options (both off by default = plain SRCNN)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.tanh_out = tanh_out

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        if self.tanh_out:
            x = torch.tanh(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick self-test: python src/model.py
    model = SRCNN()
    x = torch.rand(2, 3, 224, 224)          # a fake batch: 2 RGB images of 224x224
    y = model(x)
    print("Input shape: ", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Parameters:  ", count_params(model))
