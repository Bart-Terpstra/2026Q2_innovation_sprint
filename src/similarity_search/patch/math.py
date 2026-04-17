"""Numerical helpers for patch similarity operations.

This module provides embedding normalization and nearest-neighbor upsampling
used by patch-level similarity computations and visual overlays.
"""

import numpy as np


def normalize_embedding(x: np.ndarray, order_norm: int = 2, offset: float = 1e-8) -> np.ndarray:
    """Normalize row-wise embedding matrix to unit norm."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Unexpected matrix dimensionality {x.ndim}, expected 1 or 2")

    x_norm = np.linalg.norm(x, ord=order_norm, axis=1, keepdims=True)
    return x / (x_norm + offset)


def upsample_nearest(x: np.ndarray, h: int, w: int, patchsize: int) -> np.ndarray:
    """Nearest-neighbor upsample for 2D/3D patch maps."""
    if x.ndim == 2:
        return x.repeat(patchsize, 0).repeat(patchsize, 1)
    if x.ndim == 3:
        channels = x.shape[-1]
        return x.repeat(patchsize, 0).repeat(patchsize, 1).reshape(h, w, channels)
    raise ValueError("Unsupported ndim for upsample")
