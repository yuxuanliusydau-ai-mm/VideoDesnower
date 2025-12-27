from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform_conv import DeformableConv2d


@dataclass(frozen=True)
class MSBlockConfig:
    channels: int
    kernel_size: int = 3
    padding: int = 1


class MSBlock(nn.Module):
    def __init__(self, cfg: MSBlockConfig):
        super().__init__()
        c = cfg.channels
        k = cfg.kernel_size
        p = cfg.padding

        self.conv = nn.Conv2d(c, c, kernel_size=k, padding=p)
        self.offset_conv = nn.Conv2d(c, 2 * k * k, kernel_size=k, padding=p)
        self.deform = DeformableConv2d(c, c, kernel_size=k, padding=p)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")

        lx = self.conv(x)
        off = self.offset_conv(x)
        gx = self.deform(x, off)

        w = self.pool(x).flatten(1)
        w = F.softmax(self.fc(w), dim=-1)
        wl = w[:, 0].view(-1, 1, 1, 1)
        wg = w[:, 1].view(-1, 1, 1, 1)

        out = F.relu(wl * lx + wg * gx, inplace=True)
        return out
