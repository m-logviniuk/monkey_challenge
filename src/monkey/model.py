"""Compact U-Net density regressor with a swappable decoder head.

The backbone is a small 2D U-Net (three downsampling stages). It emits one
logit map per class; the sigmoid of that map is the predicted cell-density /
confidence heatmap that peak detection reads. The final decoder head is
either a 3x3 convolution (``head='conv'``) or a per-pixel Kolmogorov-Arnold
layer (``head='kan'``); the two are parameter-matched so a KAN-vs-conv
ablation is fair. Only the head is swapped, so the rest of the network is
identical between the two variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    BASE_CHANNELS,
    KAN_GRID_SIZE,
    KAN_SPLINE_ORDER,
    N_CLASSES,
)
from .kan import KANLinear


def _groups(channels: int) -> int:
    return min(8, max(1, channels // 4))


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvHead(nn.Module):
    """3x3 conv + norm + 1x1 projection to the class logits."""

    def __init__(self, channels: int, n_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.project = nn.Conv2d(channels, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm(self.conv(x)))
        return self.project(x)


class KANHead(nn.Module):
    """Per-pixel KAN channel mixing + norm + 1x1 projection to class logits.

    Parameter-matched to :class:`ConvHead`: with ``in_ch == out_ch == C`` a
    3x3 conv holds ``9 * C^2`` weights, while a KANLinear holds
    ``C^2 * (grid_size + spline_order) + 2 * C^2`` weights, so
    ``grid_size + spline_order + 2 == 9`` makes the two heads equal. The
    ``InstanceNorm2d`` keeps the KANLinear input inside its spline grid and is
    affine-free, so it adds no parameters.
    """

    def __init__(self, channels: int, n_classes: int,
                 grid_size: int = KAN_GRID_SIZE,
                 spline_order: int = KAN_SPLINE_ORDER):
        super().__init__()
        self.pre_norm = nn.InstanceNorm2d(channels, affine=False)
        self.kan = KANLinear(channels, channels, grid_size=grid_size,
                             spline_order=spline_order)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.project = nn.Conv2d(channels, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xk = self.pre_norm(x)
        xk = xk.permute(0, 2, 3, 1).reshape(b * h * w, c)
        xk = self.kan(xk)
        xk = xk.reshape(b, h, w, c).permute(0, 3, 1, 2)
        xk = F.gelu(self.norm(xk))
        return self.project(xk)


def build_head(head: str, channels: int, n_classes: int) -> nn.Module:
    """Return the decoder head for ``head`` in {'conv', 'kan'}."""
    if head == "conv":
        return ConvHead(channels, n_classes)
    if head == "kan":
        return KANHead(channels, n_classes)
    raise ValueError(f"unknown head {head!r}; expected 'conv' or 'kan'.")


class UNetDensity(nn.Module):
    """Three-stage 2D U-Net emitting per-class density logits.

    ``forward`` returns logits of shape ``[B, n_classes, H, W]``; apply a
    sigmoid to obtain the density / confidence heatmap.
    """

    def __init__(self, in_ch: int = 3, n_classes: int = N_CLASSES,
                 c: int = BASE_CHANNELS, head: str = "conv"):
        super().__init__()
        self.head_kind = head
        c2, c3, c4 = c * 2, c * 4, c * 8

        self.enc1 = ConvBlock(in_ch, c)
        self.enc2 = ConvBlock(c, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.dec3 = ConvBlock(c4 + c3, c3)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=False)
        self.dec1 = ConvBlock(c2 + c, c)

        self.head = build_head(head, c, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm) and m.weight is not None:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def count_params(model: nn.Module) -> dict:
    """Return total and head parameter counts.

    ``kan`` is the count of genuine KAN spline/scaler parameters when the KAN
    head is used (0 otherwise); it is a subset of ``head``.
    """
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    kan = 0
    for m in model.modules():
        if isinstance(m, KANLinear):
            kan += m.spline_weight.numel() + m.spline_scaler.numel()
    return {"total": total, "head": head, "kan": kan}
