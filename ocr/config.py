"""Training/inference configuration, loadable from YAML with CLI overrides."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    # --- data ---
    data_dir: str = "data"
    exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
    val_ratio: float = 0.1
    seed: int = 42

    # --- image / preprocessing ---
    img_h: int = 48          # must be divisible by 16
    img_w: int = 224         # fixed network width
    channels: int = 1        # 1 = grayscale, 3 = RGB
    # "pad": aspect-preserving resize to img_h then right-pad to img_w (general).
    # "stretch": plain resize straight to img_w x img_h (no padding) — simplest to
    #   deploy, ideal when the source size is fixed (e.g. data_1's 500x150).
    resize_mode: str = "pad"

    # --- model ---
    # Conv channel widths of the 4 backbone stages — the main size knob.
    # (64,128,256,512)=~8.7M params (~35MB); see configs/*_10mb.yaml / *_5mb.yaml.
    stage_channels: tuple[int, int, int, int] = (64, 128, 256, 512)
    hidden: int = 256        # BiLSTM hidden size
    lstm_layers: int = 2
    dropout: float = 0.25

    # --- training ---
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 5
    grad_clip: float = 5.0
    amp: bool = True
    num_workers: int = 2
    early_stop_patience: int = 40

    # --- augmentation ---
    # Geometric jitter. Set aug_rotate/aug_shear to 0 for orientation-sensitive
    # captchas (e.g. directional arrows) where rotation would corrupt the label.
    augment: bool = True
    aug_rotate: float = 4.0       # max rotation in degrees
    aug_translate: float = 0.06   # max fraction of width/height
    aug_scale: float = 0.1        # scale jitter is (1-x, 1+x)
    aug_shear: float = 8.0        # max shear in degrees
    aug_photometric: bool = True  # brightness/contrast/blur/noise (orientation-safe)

    # --- output ---
    out_dir: str = "runs/exp"

    @classmethod
    def load(cls, yaml_path: str | None = None, overrides: dict | None = None) -> "Config":
        cfg = cls()
        valid = {f.name for f in dataclasses.fields(cls)}
        for source, data in (("yaml", _read_yaml(yaml_path)), ("cli", overrides or {})):
            for k, v in data.items():
                if v is None:
                    continue
                if k not in valid:
                    raise KeyError(f"Unknown config key from {source}: {k!r}")
                default = getattr(cls, k, None)
                if isinstance(default, tuple) and isinstance(v, list):
                    v = tuple(v)
                setattr(cfg, k, v)
        if cfg.img_h % 16 != 0:
            raise ValueError(f"img_h must be divisible by 16, got {cfg.img_h}")
        if cfg.img_w % 4 != 0:
            raise ValueError(f"img_w must be divisible by 4, got {cfg.img_w}")
        if cfg.channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {cfg.channels}")
        if cfg.resize_mode not in ("pad", "stretch"):
            raise ValueError(f"resize_mode must be 'pad' or 'stretch', got {cfg.resize_mode!r}")
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _read_yaml(path: str | None) -> dict:
    if not path:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or {}
