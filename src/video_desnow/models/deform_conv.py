from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_kernel_grid(kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = kernel_size
    r = k // 2
    ys = torch.arange(-r, r + 1, device=device, dtype=dtype)
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
    return grid


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Only odd kernel_size is supported.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x as BCHW, got {tuple(x.shape)}")
        if offset.dim() != 4:
            raise ValueError(f"Expected offset as BCHW, got {tuple(offset.shape)}")

        b, c, h, w = x.shape
        k = self.kernel_size
        kp = k * k

        if offset.shape[0] != b or offset.shape[2] != h or offset.shape[3] != w or offset.shape[1] != 2 * kp:
            raise ValueError(
                f"offset must be (B, {2*kp}, H, W), got {tuple(offset.shape)}"
            )

        x_pad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="constant", value=0.0)
        _, _, hp, wp = x_pad.shape

        dtype = x.dtype
        device = x.device

        base_y = torch.arange(h, device=device, dtype=dtype) + self.padding
        base_x = torch.arange(w, device=device, dtype=dtype) + self.padding
        yy, xx = torch.meshgrid(base_y, base_x, indexing="ij")
        base = torch.stack([yy, xx], dim=-1)

        kgrid = _make_kernel_grid(k, device, dtype)
        ky = kgrid[:, 0].view(kp, 1, 1)
        kx = kgrid[:, 1].view(kp, 1, 1)

        coords = torch.empty((b, kp, h, w, 2), device=device, dtype=dtype)
        coords[..., 0] = base[..., 0].unsqueeze(0).unsqueeze(0) + ky
        coords[..., 1] = base[..., 1].unsqueeze(0).unsqueeze(0) + kx

        off = offset.view(b, kp, 2, h, w).permute(0, 1, 3, 4, 2)
        coords = coords + off

        gy = coords[..., 0] * (2.0 / max(hp - 1, 1)) - 1.0
        gx = coords[..., 1] * (2.0 / max(wp - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1)

        grid = grid.reshape(b * kp, h, w, 2)
        x_rep = x_pad.repeat_interleave(kp, dim=0)

        sampled = F.grid_sample(
            x_rep,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.view(b, kp, c, h, w).permute(0, 2, 1, 3, 4).reshape(b, c * kp, h, w)

        w_flat = self.weight.view(self.out_channels, self.in_channels * kp, 1, 1)
        out = F.conv2d(sampled, w_flat, bias=self.bias, stride=1, padding=0)
        return out
