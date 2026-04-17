"""Feature-extraction utilities for patch-based similarity analysis.

This module converts images into per-patch embeddings and returns a normalized
state dictionary used by visualization and matching components.
"""

from typing import Any

import torch
from PIL import Image

from similarity_search.image.processing import crop_to_patchsize, np_array, preprocess_image
from similarity_search.patch.math import normalize_embedding


def build_state(
    pil_img: Image.Image,
    model: torch.nn.Module,
    patchsize: int,
    device: torch.device,
) -> dict[str, Any]:
    """Build patch embedding state for one image and one ViT-like model."""
    img_tensor = preprocess_image(pil_img, patchsize, device)
    img_np = np_array(crop_to_patchsize(pil_img, patchsize=patchsize))

    _, _, h, w = img_tensor.shape
    rows, cols = h // patchsize, w // patchsize

    with torch.no_grad():
        y = model(img_tensor)
        hs = y.last_hidden_state.squeeze(0).detach().cpu().numpy()

    n_patches = rows * cols
    patch_embs = hs[-n_patches:, :].reshape(rows, cols, -1)
    x = patch_embs.reshape(-1, patch_embs.shape[-1])
    xn = normalize_embedding(x)

    return {
        "img": img_np,
        "ps": patchsize,
        "h": h,
        "w": w,
        "rows": rows,
        "cols": cols,
        "X": x,
        "Xn": xn,
    }
