from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kformer import KFormer2D, KFormer2DConfig
from .msblock import MSBlock, MSBlockConfig


@dataclass(frozen=True)
class VideoDesnowerNetConfig:
    frames: int = 5
    image_size: int = 224
    channels: int = 64
    num_heads: int = 4
    mlp_ratio: float = 2.0
    k_max: int = 12
    k_min: int = 1
    knn_mode: str = "softmax"
    knn_temperature: float = 1.0
    dropout: float = 0.0
    patch_size_kformer: int = 4
    kformer_depth: int = 1
    use_residual_to_center: bool = True


class VideoDesnowerNet(nn.Module):
    def __init__(self, cfg: VideoDesnowerNetConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = cfg.frames * 3
        c = cfg.channels

        self.in_conv = nn.Conv2d(in_ch, c, kernel_size=3, padding=1)

        self.ms1 = MSBlock(MSBlockConfig(channels=c))
        self.down1 = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)

        self.ms2 = MSBlock(MSBlockConfig(channels=c))
        self.kformer1 = KFormer2D(
            KFormer2DConfig(
                channels=c,
                patch_size=cfg.patch_size_kformer,
                depth=cfg.kformer_depth,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                k_max=cfg.k_max,
                k_min=cfg.k_min,
                knn_mode=cfg.knn_mode,
                knn_temperature=cfg.knn_temperature,
            )
        )
        self.down2 = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)

        self.ms3 = MSBlock(MSBlockConfig(channels=c))
        self.kformer2 = KFormer2D(
            KFormer2DConfig(
                channels=c,
                patch_size=cfg.patch_size_kformer,
                depth=cfg.kformer_depth,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                k_max=cfg.k_max,
                k_min=cfg.k_min,
                knn_mode=cfg.knn_mode,
                knn_temperature=cfg.knn_temperature,
            )
        )

        self.up1 = nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)
        self.ms4 = MSBlock(MSBlockConfig(channels=c))

        self.up2 = nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)
        self.ms5 = MSBlock(MSBlockConfig(channels=c))

        self.out_conv = nn.Conv2d(c, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected (B,T,3,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if t != self.cfg.frames:
            raise ValueError(f"Configured frames={self.cfg.frames}, got {t}")
        if c != 3:
            raise ValueError(f"Expected RGB input (c=3), got {c}")
        if h != self.cfg.image_size or w != self.cfg.image_size:
            raise ValueError(f"Expected image_size={self.cfg.image_size}, got {(h,w)}")

        center = x[:, t // 2]
        x = x.reshape(b, t * c, h, w)

        f0 = self.in_conv(x)

        f1 = self.ms1(f0)
        f2 = self.down1(f1)

        f2 = self.ms2(f2)
        f2 = f2 + self.kformer1(f2)

        f3 = self.down2(f2)
        f3 = self.ms3(f3)
        f3 = f3 + self.kformer2(f3)

        d2 = self.up1(f3) + f2
        d2 = self.ms4(d2)

        d1 = self.up2(d2) + f1
        d1 = self.ms5(d1)

        out = self.out_conv(d1)

        if self.cfg.use_residual_to_center:
            out = (out + center).clamp(0.0, 1.0)
        return out
