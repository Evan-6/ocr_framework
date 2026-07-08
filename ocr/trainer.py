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
from .dataset import CaptchaDataset, collate_ctc
from .splits import load_or_create_split
from .decoder import ctc_greedy_decode
from .metrics import compute_metrics, detailed_metrics
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
def _collect(model, loader, charset, device, ctc):
    """Return (preds, labels, mean_loss) over a loader — used for final reporting."""
    model.eval()
    preds, labels = [], []
    loss_sum, n_batches = 0.0, 0
    for images, targets, target_lengths, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds.extend(ctc_greedy_decode(logits, charset))
        labels.extend(batch_labels)
        log_probs = F.log_softmax(logits.float(), dim=-1).permute(1, 0, 2)
        input_lengths = torch.full((images.size(0),), log_probs.size(0),
                                   dtype=torch.long, device=device)
        loss_sum += ctc(log_probs, targets.to(device), input_lengths,
                        target_lengths.to(device)).item()
        n_batches += 1
    return preds, labels, loss_sum / max(1, n_batches)


@torch.no_grad()
def validate(model, loader, charset, device, ctc=None) -> dict:
    model.eval()
    preds, labels = [], []
    loss_sum = 0.0
    n_batches = 0
    for images, targets, target_lengths, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds.extend(ctc_greedy_decode(logits, charset))
        labels.extend(batch_labels)
        
        if ctc is not None:
            targets = targets.to(device)
            log_probs = F.log_softmax(logits.float(), dim=-1).permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = ctc(log_probs, targets, input_lengths, target_lengths.to(device))
            loss_sum += loss.item()
            n_batches += 1
            
    m = compute_metrics(preds, labels)
    if ctc is not None:
        m["val_loss"] = loss_sum / max(1, n_batches)
    return m


def train(cfg: Config) -> Path:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = scan_dataset(cfg.data_dir, cfg.exts)
    train_samples, val_samples, test_samples, split_hash = load_or_create_split(
        samples, cfg.val_ratio, cfg.test_ratio, cfg.seed, cfg.data_dir, cfg.split_file
    )
    charset = Charset.from_labels([lb for _, lb in train_samples + val_samples])
    unseen = sorted(set("".join(lb for _, lb in test_samples)) - set(charset.chars))
    if unseen:
        print(f"[warn] test set contains chars unseen in train/val (always wrong): {unseen}")
    max_len = max(len(lb) for _, lb in samples)
    n_timesteps = cfg.img_w // 4
    print(f"[data] {len(samples)} samples ({len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test), "
          f"charset={charset.chars!r} ({charset.num_classes - 1} chars), "
          f"max label len={max_len}, CTC timesteps={n_timesteps}")
    if n_timesteps < 2 * max_len + 1:
        print(f"[warn] timesteps ({n_timesteps}) < 2*max_len+1 ({2 * max_len + 1}); "
              f"consider increasing img_w")

    save_meta(out_dir / "model_meta.json", charset, cfg)
    
    meta_path = out_dir / "model_meta.json"
    with open(meta_path, "r") as f:
        meta_dict = json.load(f)
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        git_commit = "unknown"
    meta_dict.update({
        "git_commit": git_commit,
        "split_hash": split_hash,
        "seed": cfg.seed,
        "environment": f"torch={torch.__version__}",
    })
    with open(meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2)
        
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

    best_score = float("inf") if cfg.select_mode == "min" else -float("inf")
    best_acc, best_epoch = 0.0, -1
    t0 = time.time()
    # clear metrics.jsonl
    (out_dir / "metrics.jsonl").write_text("", encoding="utf-8")
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

        m = validate(model, val_loader, charset, device, ctc)
        lr_now = scheduler.get_last_lr()[0]
        
        if cfg.select_metric == "val_loss":
            score = m.get("val_loss", float("inf"))
        elif cfg.select_metric == "seq_acc":
            score = m["seq_acc"] + 1e-3 * m["char_acc"]
        else:
            score = m.get(cfg.select_metric, 0.0)

        if cfg.select_mode == "min":
            improved = score < best_score
        else:
            improved = score > best_score

        if improved:
            best_score, best_acc, best_epoch = score, m["seq_acc"], epoch
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                        "charset": charset.chars, "epoch": epoch,
                        "val_seq_acc": m["seq_acc"]}, out_dir / "best.pt")
        print(f"epoch {epoch:3d}/{cfg.epochs} | loss {loss_sum / max(1, n_batches):.4f} | "
              f"val_loss {m.get('val_loss', 0.0):.4f} | "
              f"val seq_acc {m['seq_acc']:.4f} char_acc {m['char_acc']:.4f} | "
              f"lr {lr_now:.2e} | {time.time() - t0:.0f}s{' *' if improved else ''}")
              
        epoch_metrics = {
            "epoch": epoch,
            "loss": loss_sum / max(1, n_batches),
            "val_loss": m.get("val_loss", 0.0),
            "val_seq_acc": m["seq_acc"],
            "val_char_acc": m["char_acc"]
        }
        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(epoch_metrics) + "\n")

        if epoch - best_epoch >= cfg.early_stop_patience:
            print(f"[early stop] no improvement for {cfg.early_stop_patience} epochs")
            break

    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "charset": charset.chars, "epoch": epoch}, out_dir / "last.pt")

    # Final report on the *best* checkpoint (not the last), including the held-out
    # test set which never influenced training or model selection.
    if (out_dir / "best.pt").exists():
        best_ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model"])

    final_metrics = {"best_epoch": best_epoch, "split_hash": split_hash}

    vp, vl, v_loss = _collect(model, val_loader, charset, device, ctc)
    vm = compute_metrics(vp, vl)
    final_metrics["val"] = {"seq_acc": vm["seq_acc"], "char_acc": vm["char_acc"],
                            "loss": v_loss, "n": vm["n"]}

    if test_samples:
        test_ds = CaptchaDataset(test_samples, charset, cfg, training=False)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, collate_fn=collate_ctc)
        tp, tl, t_loss = _collect(model, test_loader, charset, device, ctc)
        tm = compute_metrics(tp, tl)
        final_metrics["test"] = {"seq_acc": tm["seq_acc"], "char_acc": tm["char_acc"],
                                 "loss": t_loss, "n": tm["n"], **detailed_metrics(tp, tl)}
        errors = [{"path": path.name, "gt": t, "pred": p}
                  for (path, _), p, t in zip(test_samples, tp, tl) if p != t]
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"[test] seq_acc {tm['seq_acc']:.4f} char_acc {tm['char_acc']:.4f} "
              f"(n={tm['n']}, {len(errors)} wrong)")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"[done] best val seq_acc {final_metrics['val']['seq_acc']:.4f} @ epoch {best_epoch} "
          f"-> {out_dir / 'best.pt'}")
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
