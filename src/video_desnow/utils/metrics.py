from __future__ import annotations

import math
import numpy as np


def psnr(x: np.ndarray, y: np.ndarray, data_range: float = 255.0) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim(x: np.ndarray, y: np.ndarray, data_range: float = 255.0) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape}")

    if x.ndim == 2:
        x = x[..., None]
        y = y[..., None]

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    vals = []
    for ch in range(x.shape[-1]):
        xi = x[..., ch]
        yi = y[..., ch]

        mu_x = float(xi.mean())
        mu_y = float(yi.mean())
        var_x = float(xi.var())
        var_y = float(yi.var())
        cov = float(((xi - mu_x) * (yi - mu_y)).mean())

        num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
        den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        vals.append(num / (den + 1e-12))

    return float(np.mean(vals))
