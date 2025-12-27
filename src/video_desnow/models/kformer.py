from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .knn import AdaptiveKNNConfig, adaptive_knn_aggregate


@dataclass(frozen=True)
class AdaptiveKValuesNetConfig:
    in_dim: int
    hidden_dim: int = 128
    num_ks: int = 3
    k_max: int = 12
    k_min: int = 1


class AdaptiveKValuesNet(nn.Module):
    def __init__(self, cfg: AdaptiveKValuesNetConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.num_ks),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected tokens (B,N,C), got {tuple(x.shape)}")
        pooled = x.mean(dim=1)
        ratio = self.net(pooled)
        k_cont = self.cfg.k_min + ratio * (self.cfg.k_max - self.cfg.k_min)
        k_int = torch.round(k_cont).clamp(self.cfg.k_min, self.cfg.k_max).to(dtype=torch.int64)
        return k_cont, k_int


@dataclass(frozen=True)
class AdaptiveWeightNetConfig:
    in_dim: int
    hidden_dim: int = 128
    num_branches: int = 3


class AdaptiveWeightNet(nn.Module):
    def __init__(self, cfg: AdaptiveWeightNetConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.num_branches),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected tokens (B,N,C), got {tuple(x.shape)}")
        pooled = x.mean(dim=1)
        return F.softmax(self.net(pooled), dim=-1)


@dataclass(frozen=True)
class KNNFeedForwardConfig:
    dim: int
    hidden_dim: int
    k_max: int = 12
    k_min: int = 1
    knn_mode: str = "softmax"
    knn_temperature: float = 1.0
    ste_alpha: float = 12.0
    dropout: float = 0.0
    k_hidden_dim: int = 128
    w_hidden_dim: int = 128


class KNNFeedForward(nn.Module):
    def __init__(self, cfg: KNNFeedForwardConfig):
        super().__init__()
        self.cfg = cfg

        self.fc1 = nn.Linear(cfg.dim, cfg.hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=cfg.dropout)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.dim)
        self.drop2 = nn.Dropout(p=cfg.dropout)

        self.k_net = AdaptiveKValuesNet(
            AdaptiveKValuesNetConfig(
                in_dim=cfg.dim,
                hidden_dim=cfg.k_hidden_dim,
                num_ks=3,
                k_max=cfg.k_max,
                k_min=cfg.k_min,
            )
        )
        self.w_net = AdaptiveWeightNet(
            AdaptiveWeightNetConfig(
                in_dim=cfg.dim,
                hidden_dim=cfg.w_hidden_dim,
                num_branches=3,
            )
        )

        self.knn_cfg = AdaptiveKNNConfig(
            k_max=cfg.k_max,
            k_min=cfg.k_min,
            mode=cfg.knn_mode,
            temperature=cfg.knn_temperature,
            ste_alpha=cfg.ste_alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B,N,C), got {tuple(x.shape)}")

        k_cont, _ = self.k_net(x)
        w = self.w_net(x)

        h = self.fc1(x)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)

        outs = []
        for i in range(3):
            outs.append(adaptive_knn_aggregate(h, k_cont[:, i], self.knn_cfg))

        y = outs[0] * w[:, 0].view(-1, 1, 1) + outs[1] * w[:, 1].view(-1, 1, 1) + outs[2] * w[:, 2].view(-1, 1, 1)
        return y


@dataclass(frozen=True)
class KFormerBlockConfig:
    dim: int
    num_heads: int = 4
    mlp_ratio: float = 2.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    ffn_dropout: float = 0.0
    k_max: int = 12
    k_min: int = 1
    knn_mode: str = "softmax"
    knn_temperature: float = 1.0
    ste_alpha: float = 12.0


class KFormerBlock(nn.Module):
    def __init__(self, cfg: KFormerBlockConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.dim,
            num_heads=cfg.num_heads,
            dropout=cfg.attn_dropout,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(p=cfg.proj_dropout)

        self.norm2 = nn.LayerNorm(cfg.dim)
        self.ffn = KNNFeedForward(
            KNNFeedForwardConfig(
                dim=cfg.dim,
                hidden_dim=int(cfg.dim * cfg.mlp_ratio),
                k_max=cfg.k_max,
                k_min=cfg.k_min,
                knn_mode=cfg.knn_mode,
                knn_temperature=cfg.knn_temperature,
                ste_alpha=cfg.ste_alpha,
                dropout=cfg.ffn_dropout,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.proj_drop(attn_out)

        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


@dataclass(frozen=True)
class KFormer2DConfig:
    channels: int
    patch_size: int = 4
    depth: int = 1
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    k_max: int = 12
    k_min: int = 1
    knn_mode: str = "softmax"
    knn_temperature: float = 1.0
    ste_alpha: float = 12.0


class KFormer2D(nn.Module):
    def __init__(self, cfg: KFormer2DConfig):
        super().__init__()
        self.cfg = cfg
        ps = cfg.patch_size
        c = cfg.channels

        self.patch = nn.Conv2d(c, c, kernel_size=ps, stride=ps)
        self.unpatch = nn.ConvTranspose2d(c, c, kernel_size=ps, stride=ps)

        self.blocks = nn.ModuleList(
            [
                KFormerBlock(
                    KFormerBlockConfig(
                        dim=c,
                        num_heads=cfg.num_heads,
                        mlp_ratio=cfg.mlp_ratio,
                        attn_dropout=cfg.dropout,
                        proj_dropout=cfg.dropout,
                        ffn_dropout=cfg.dropout,
                        k_max=cfg.k_max,
                        k_min=cfg.k_min,
                        knn_mode=cfg.knn_mode,
                        knn_temperature=cfg.knn_temperature,
                        ste_alpha=cfg.ste_alpha,
                    )
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
        b, c, h, w = x.shape
        ps = self.cfg.patch_size
        if h % ps != 0 or w % ps != 0:
            raise ValueError(f"H/W must be divisible by patch_size={ps}, got {(h,w)}")

        p = self.patch(x)
        ph, pw = p.shape[-2], p.shape[-1]
        tokens = p.flatten(2).transpose(1, 2)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        p2 = tokens.transpose(1, 2).reshape(b, c, ph, pw)
        y = self.unpatch(p2)
        return y
