"""Dataset and CTC collate."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .charset import Charset
from .transforms import add_noise, build_augment, preprocess





class CaptchaDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, str]], charset: Charset, cfg, training: bool):
        valid_samples = []
        for p, label in samples:
            try:
                charset.encode(label)
                valid_samples.append((p, label))
            except KeyError as e:
                print(f"[dataset] Warning: skipped {p.name} (label {label!r}) due to unseen character.")
                
        self.samples = valid_samples
        self.charset = charset
        self.cfg = cfg
        self.training = training
        
        # ─── 💡 修正後的安全判斷 💡 ───
        # 只要 training=True 且 yaml 有開啟幾何或光度增強，就直接啟動
        has_any_aug = getattr(cfg, "augment", False) or cfg.aug_rotate or cfg.aug_translate or cfg.aug_scale or cfg.aug_shear or cfg.aug_photometric
        
        self.augment = build_augment(cfg) if (training and has_any_aug) else None
        self.noise = training and cfg.aug_photometric

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path)
        if self.augment is not None:
            img = self.augment(img.convert("RGB"))
        x = torch.from_numpy(preprocess(img, self.cfg.img_h, self.cfg.img_w,
                                        self.cfg.channels, self.cfg.resize_mode))
        if self.noise:
            x = add_noise(x)
        target = torch.tensor(self.charset.encode(label), dtype=torch.long)
        return x, target, label


def collate_ctc(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    target_lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    labels = [b[2] for b in batch]
    return images, targets, target_lengths, labels
