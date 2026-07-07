"""CRNN: CNN backbone -> BiLSTM neck -> linear CTC head.

Kept modular so the backbone/neck can be swapped later without touching the
training pipeline. All ops (Conv/BN/ReLU/MaxPool/LSTM/Linear) export cleanly
to ONNX and are fully supported by ONNX Runtime C++.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CNNBackbone(nn.Module):
    """Downsamples height x16 and width x4; final conv squeezes height to 1.

    Channel widths of the four stages are configurable (``stage_channels``),
    which is the main knob for scaling model size. Defaults reproduce the
    original VGG-style ~8.7M-param backbone.
    """

    def __init__(self, channels: int, img_h: int,
                 stage_channels: tuple[int, int, int, int] = (64, 128, 256, 512)):
        super().__init__()
        assert img_h % 16 == 0, "img_h must be divisible by 16"
        c1, c2, c3, c4 = stage_channels
        self.features = nn.Sequential(
            _conv_block(channels, c1),
            nn.MaxPool2d(2, 2),                    # H/2,  W/2
            _conv_block(c1, c2),
            nn.MaxPool2d(2, 2),                    # H/4,  W/4
            _conv_block(c2, c3),
            _conv_block(c3, c3),
            nn.MaxPool2d((2, 1), (2, 1)),          # H/8,  W/4
            _conv_block(c3, c4),
            _conv_block(c4, c4),
            nn.MaxPool2d((2, 1), (2, 1)),          # H/16, W/4
            nn.Conv2d(c4, c4, (img_h // 16, 1), bias=False),  # H -> 1
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.out_channels = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)          # (B, c4, 1, W/4)
        return f.squeeze(2).permute(0, 2, 1)  # (B, T, c4)


class BiLSTMNeck(nn.Module):
    """Stacked single-layer BiLSTMs with explicit Dropout in between.

    Deliberately not nn.LSTM(num_layers=2, dropout=...): cuDNN's internal
    dropout state crashes at process teardown on some Windows/CUDA builds,
    and explicit layers export to cleaner ONNX.
    """

    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstms = nn.ModuleList(
            nn.LSTM(in_dim if i == 0 else hidden * 2, hidden,
                    bidirectional=True, batch_first=True)
            for i in range(layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lstm in enumerate(self.lstms):
            x, _ = lstm(x)
            if i < len(self.lstms) - 1:
                x = self.dropout(x)
        return x


class CRNN(nn.Module):
    def __init__(self, channels: int, img_h: int, num_classes: int,
                 hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.25,
                 stage_channels: tuple[int, int, int, int] = (64, 128, 256, 512)):
        super().__init__()
        self.backbone = CNNBackbone(channels, img_h, stage_channels)
        self.neck = BiLSTMNeck(self.backbone.out_channels, hidden, lstm_layers, dropout)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.neck(self.backbone(x)))  # (B, T, num_classes) logits


def build_model(cfg, num_classes: int) -> CRNN:
    return CRNN(cfg.channels, cfg.img_h, num_classes,
                hidden=cfg.hidden, lstm_layers=cfg.lstm_layers, dropout=cfg.dropout,
                stage_channels=tuple(cfg.stage_channels))
