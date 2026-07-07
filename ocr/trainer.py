"""Training loop: CTC loss, AMP, warmup+cosine schedule, early stopping."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .charset import Charset, save_meta, scan_dataset
from .config import Config
from .dataset import CaptchaDataset, collate_ctc, split_samples
from .decoder import ctc_greedy_decode
from .metrics import compute_metrics
from .model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: Config, charset: Charset, train_samples, val_samples):
    train_ds = CaptchaDataset(train_samples, charset, cfg, training=True)
    val_ds = CaptchaDataset(val_samples, charset, cfg, training=False)
    kw = dict(num_workers=cfg.num_workers, collate_fn=collate_ctc,
              persistent_workers=cfg.num_workers > 0, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=len(train_ds) > cfg.batch_size, **kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **kw)
    return train_loader, val_loader


def make_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    warmup = max(1, cfg.warmup_epochs * steps_per_epoch)
    total = max(warmup + 1, cfg.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / (total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, loader, charset, device) -> dict:
    model.eval()
    preds, labels = [], []
    for images, _, _, batch_labels in loader:
        logits = model(images.to(device, non_blocking=True))
        preds.extend(ctc_greedy_decode(logits, charset))
        labels.extend(batch_labels)
    return compute_metrics(preds, labels)


def train(cfg: Config) -> Path:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = scan_dataset(cfg.data_dir, cfg.exts)
    charset = Charset.from_labels([lb for _, lb in samples])
    train_samples, val_samples = split_samples(samples, cfg.val_ratio, cfg.seed)
    max_len = max(len(lb) for _, lb in samples)
    n_timesteps = cfg.img_w // 4
    print(f"[data] {len(samples)} samples ({len(train_samples)} train / {len(val_samples)} val), "
          f"charset={charset.chars!r} ({charset.num_classes - 1} chars), "
          f"max label len={max_len}, CTC timesteps={n_timesteps}")
    if n_timesteps < 2 * max_len + 1:
        print(f"[warn] timesteps ({n_timesteps}) < 2*max_len+1 ({2 * max_len + 1}); "
              f"consider increasing img_w")

    save_meta(out_dir / "model_meta.json", charset, cfg)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    train_loader, val_loader = build_loaders(cfg, charset, train_samples, val_samples)
    model = build_model(cfg, charset.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] CRNN {n_params / 1e6:.2f}M params, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg, len(train_loader))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ctc = torch.nn.CTCLoss(blank=Charset.BLANK, zero_infinity=True)

    best_score, best_acc, best_epoch = -1.0, 0.0, -1
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)                      # (B, T, C)
            log_probs = F.log_softmax(logits.float(), dim=-1).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((images.size(0),), log_probs.size(0),
                                       dtype=torch.long, device=device)
            loss = ctc(log_probs, targets, input_lengths, target_lengths.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev_scale:  # skip scheduler if AMP skipped optimizer
                scheduler.step()
            loss_sum += loss.item()
            n_batches += 1

        m = validate(model, val_loader, charset, device)
        lr_now = scheduler.get_last_lr()[0]
        # Rank by seq_acc, break ties with char_acc so progress still registers
        # while seq_acc is stuck at 0 (common early on / on hard small sets).
        score = m["seq_acc"] + 1e-3 * m["char_acc"]
        improved = score > best_score
        if improved:
            best_score, best_acc, best_epoch = score, m["seq_acc"], epoch
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                        "charset": charset.chars, "epoch": epoch,
                        "val_seq_acc": m["seq_acc"]}, out_dir / "best.pt")
        print(f"epoch {epoch:3d}/{cfg.epochs} | loss {loss_sum / max(1, n_batches):.4f} | "
              f"val seq_acc {m['seq_acc']:.4f} char_acc {m['char_acc']:.4f} | "
              f"lr {lr_now:.2e} | {time.time() - t0:.0f}s{' *' if improved else ''}")

        if epoch - best_epoch >= cfg.early_stop_patience:
            print(f"[early stop] no improvement for {cfg.early_stop_patience} epochs")
            break

    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "charset": charset.chars, "epoch": epoch}, out_dir / "last.pt")
    print(f"[done] best val seq_acc {best_acc:.4f} @ epoch {best_epoch} -> {out_dir / 'best.pt'}")
    return out_dir / "best.pt"


def load_checkpoint(ckpt_path: str | Path, device: str = "cpu"):
    """Rebuild (model, charset, cfg) from a self-contained checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = Config.load(overrides=ckpt["config"])
    charset = Charset(ckpt["charset"])
    model = build_model(cfg, charset.num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, charset, cfg
