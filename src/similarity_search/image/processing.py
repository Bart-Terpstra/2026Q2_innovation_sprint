"""Image loading and preprocessing helpers for similarity pipelines.

This module includes image I/O, patch-size cropping, NumPy conversion, and
tensor preprocessing with ImageNet normalization.
"""

from PIL import Image
from torchvision import transforms
import numpy as np
import torch


def load_image(path: str) -> Image:
    """Load an image from PATH and return a PIL RGB image."""
    return Image.open(path).convert("RGB")


def crop_to_patchsize(pil_img: Image, patchsize: int=16) -> Image:
    """Pad PIL image on right/bottom so (h,w) are multiples of `patchsize`."""
    w, h = pil_img.size
    h_crop = (h // patchsize) * patchsize
    w_crop = (w // patchsize) * patchsize
    return pil_img.crop((0, 0, w_crop, h_crop))

def np_array(pil_img: Image, dtype: np.dtype =np.uint8) -> np.ndarray:
    return np.array(pil_img, dtype)
    
def preprocess_image(pil_img: Image, patchsize: int, device: torch.device) -> torch.Tensor:
    """Crop (right/bottom) -> ToTensor -> Normalize (ImageNet stats)."""
    img_padded = crop_to_patchsize(pil_img, patchsize)
    transform = transforms.Compose([
        transforms.ToTensor(),  # CxHxW in range [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_padded).unsqueeze(0).to(device)  # (1,3,H,W)
    return tensor