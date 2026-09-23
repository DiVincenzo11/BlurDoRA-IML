"""Bayar-constrained residual branch used by BlurDoRA-IML.

Input is a 512x512 ImageNet-normalized RGB tensor. The branch first restores
RGB values to [0, 1], then produces 1024 side tokens with 256 channels each.
"""

from __future__ import annotations

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BayarConv2d(nn.Module):
    """A 5x5 RGB-to-residual convolution with constrained kernels."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=5, padding=2, bias=False)
        mask = torch.ones(5, 5)
        mask[2, 2] = 0.0
        self.register_buffer("mask", mask.view(1, 1, 5, 5))
        self.project_constraints()

    @torch.no_grad()
    def project_constraints(self) -> None:
        """Set each center coefficient to -1 and off-center sum to +1.

        Call this once after every optimizer step during training, as in the
        paper experiments. This initial release does not include that loop.
        """
        weights = self.conv.weight
        off_center = self.mask[0, 0]
        center = 2
        eps = 1e-8
        for out_channel in range(weights.shape[0]):
            for in_channel in range(weights.shape[1]):
                kernel = weights[out_channel, in_channel]
                remainder = kernel * off_center
                total = remainder.sum()
                if total.abs() < eps:
                    remainder = off_center / off_center.sum().clamp(min=eps)
                    total = remainder.sum()
                elif total < 0:
                    remainder = -remainder
                    total = -total
                remainder = remainder / total.clamp(min=eps)
                kernel.copy_(remainder)
                kernel[center, center] = -1.0

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.conv(rgb)


class ResidualBranch(nn.Module):
    """Four stride-2 Conv-BN-ReLU blocks map residuals to 32x32 side tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.bayar = BayarConv2d()
        blocks: list[nn.Module] = []
        for in_channels, out_channels in ((1, 32), (32, 64), (64, 128), (128, 256)):
            blocks.extend(
                (
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.stem = nn.Sequential(*blocks)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (3, 512, 512):
            raise ValueError("Expected ImageNet-normalized RGB [B,3,512,512]")
        with torch.amp.autocast("cuda", enabled=False):
            image = image.float()
            mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
            rgb = (image * std + mean).clamp(0.0, 1.0)
            features = self.stem(self.bayar(rgb))
        if features.shape[1:] != (256, 32, 32):
            raise RuntimeError(f"Unexpected side feature shape: {features.shape}")
        return features.flatten(2).transpose(1, 2)  # [B, 1024, 256]
