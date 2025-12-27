from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn.functional as F


def pairwise_l2_distance(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"x must be (B,N,C), got {tuple(x.shape)}")
    x2 = (x * x).sum(dim=-1, keepdim=True)
    dist = x2 + x2.transpose(1, 2) - 2.0 * (x @ x.transpose(1, 2))
    return dist.clamp_min(0.0)


def knn_topk(
    x: torch.Tensor,
    k_max: int,
    exclude_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k_max <= 0:
        raise ValueError(f"k_max must be positive, got {k_max}")
    if x.dim() != 3:
        raise ValueError(f"x must be (B,N,C), got {tuple(x.shape)}")
    b, n, _ = x.shape
    k_max = min(k_max, n - 1 if exclude_self else n)

    dist = pairwise_l2_distance(x)
    if exclude_self:
        inf = torch.finfo(dist.dtype).max
        eye = torch.eye(n, device=x.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(eye, inf)

    idx = dist.topk(k_max, dim=-1, largest=False).indices
    dist_k = dist.gather(dim=-1, index=idx)
    return idx, dist_k


@dataclass(frozen=True)
class AdaptiveKNNConfig:
    k_max: int = 12
    k_min: int = 1
    mode: Literal["mean", "softmax"] = "softmax"
    temperature: float = 1.0
    exclude_self: bool = True
    ste_alpha: float = 12.0


def adaptive_knn_aggregate(
    x: torch.Tensor,
    k_cont: torch.Tensor,
    cfg: AdaptiveKNNConfig,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"x must be (B,N,C), got {tuple(x.shape)}")
    if k_cont.dim() != 1:
        raise ValueError(f"k_cont must be (B,), got {tuple(k_cont.shape)}")

    b, n, c = x.shape
    idx, dist_k = knn_topk(x, cfg.k_max, exclude_self=cfg.exclude_self)
    batch = torch.arange(b, device=x.device)[:, None, None]
    neigh = x[batch, idx]

    ranks = torch.arange(1, cfg.k_max + 1, device=x.device, dtype=x.dtype)[None, None, :]
    k_soft = k_cont.view(b, 1, 1).to(dtype=x.dtype)

    mask_soft = torch.sigmoid(cfg.ste_alpha * (k_soft - ranks))
    k_round = torch.round(k_cont).clamp(cfg.k_min, cfg.k_max).view(b, 1, 1).to(dtype=x.dtype)
    mask_hard = (ranks <= k_round).to(dtype=x.dtype)
    mask = (mask_hard - mask_soft).detach() + mask_soft

    if cfg.mode == "mean":
        w = mask / (mask.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return (neigh * w.unsqueeze(-1)).sum(dim=2)

    if cfg.mode == "softmax":
        w = F.softmax(-dist_k / cfg.temperature, dim=-1)
        w = w * mask
        w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        return (neigh * w.unsqueeze(-1)).sum(dim=2)

    raise ValueError(f"Unsupported mode: {cfg.mode}")
