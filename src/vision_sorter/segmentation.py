"""Compact U-Net: skip connections retain spatial detail for pixel labels."""

import torch
from torch import nn


def block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU()
    )


class SorterUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = block(3, 16)
        self.middle = block(16, 32)
        self.up = block(48, 16)
        self.head = nn.Conv2d(16, 5, 1)  # background + four material classes

    def forward(self, x):
        skip = self.down(x)
        low = self.middle(nn.functional.max_pool2d(skip, 2))
        high = nn.functional.interpolate(
            low, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.head(self.up(torch.cat([skip, high], dim=1)))
