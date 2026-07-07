"""Preprocessing (aspect-preserving resize + right pad) and train-time augmentation."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


def preprocess(img: Image.Image, img_h: int, img_w: int, channels: int,
               resize_mode: str = "pad") -> np.ndarray:
    """PIL image -> float32 CHW in [-1, 1], fixed (channels, img_h, img_w).

    resize_mode="pad": resize height to img_h keeping aspect ratio, cap width at
        img_w, right-pad with black. Handles arbitrary/varying source sizes.
    resize_mode="stretch": resize straight to (img_w, img_h), no padding. Simplest
        for deployment when the source size is fixed.
    Must be mirrored exactly by any deployment (see cpp_example/).
    """
    img = img.convert("L" if channels == 1 else "RGB")
    if resize_mode == "stretch":
        img = img.resize((img_w, img_h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        canvas = arr[:, :, None] if channels == 1 else arr
    else:
        w, h = img.size
        new_w = min(img_w, max(1, round(w * img_h / h)))
        img = img.resize((new_w, img_h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if channels == 1:
            arr = arr[:, :, None]
        canvas = np.zeros((img_h, img_w, channels), dtype=np.float32)
        canvas[:, :new_w] = arr
    canvas = canvas / 127.5 - 1.0
    return canvas.transpose(2, 0, 1)  # CHW


# Backwards-compatible alias.
def resize_and_pad(img: Image.Image, img_h: int, img_w: int, channels: int) -> np.ndarray:
    return preprocess(img, img_h, img_w, channels, "pad")


def build_augment(cfg) -> T.Compose | None:
    """Geometric + photometric jitter applied on the PIL image before resize.

    Geometry and photometry are toggled independently via the config so that
    orientation-sensitive captchas can keep photometric noise while dropping
    rotation/shear. Returns None if nothing is enabled.
    """
    aug = []
    affine_kw = {}
    if cfg.aug_rotate or cfg.aug_translate or cfg.aug_scale or cfg.aug_shear:
        affine_kw["degrees"] = cfg.aug_rotate
        if cfg.aug_translate:
            affine_kw["translate"] = (cfg.aug_translate / 2, cfg.aug_translate)
        if cfg.aug_scale:
            affine_kw["scale"] = (1.0 - cfg.aug_scale, 1.0 + cfg.aug_scale)
        if cfg.aug_shear:
            affine_kw["shear"] = cfg.aug_shear
        aug.append(T.RandomApply([T.RandomAffine(**affine_kw)], p=0.7))

    if cfg.aug_photometric:
        jitter = T.ColorJitter(brightness=0.25, contrast=0.25) if cfg.channels == 1 else \
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)
        aug.append(T.RandomApply([jitter], p=0.5))
        aug.append(T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.2))

    return T.Compose(aug) if aug else None


def add_noise(x: torch.Tensor, p: float = 0.3, std: float = 0.03) -> torch.Tensor:
    if torch.rand(()) < p:
        x = (x + torch.randn_like(x) * std).clamp_(-1.0, 1.0)
    return x
