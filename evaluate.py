"""Evaluate a checkpoint.

    python evaluate.py --ckpt runs/data_2/best.pt              # val split of training data
    python evaluate.py --ckpt runs/data_2/best.pt --data-dir other_folder --split all
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ocr.charset import scan_dataset
from ocr.dataset import CaptchaDataset, collate_ctc
from ocr.splits import load_or_create_split
from ocr.decoder import ctc_greedy_decode
from ocr.metrics import compute_metrics
from ocr.trainer import load_checkpoint


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", help="defaults to the training data_dir from the checkpoint")
    ap.add_argument("--split", choices=["val", "test", "all"], default="val",
                    help="'val' or 'test' re-derives the split; 'all' uses everything")
    ap.add_argument("--show-errors", type=int, default=10, help="print up to N wrong predictions")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, charset, cfg = load_checkpoint(args.ckpt, device)
    samples = scan_dataset(args.data_dir or cfg.data_dir, cfg.exts)
    if args.split in ("val", "test"):
        train_s, val_s, test_s, _ = load_or_create_split(
            samples, cfg.val_ratio, getattr(cfg, "test_ratio", 0.1), cfg.seed, args.data_dir or cfg.data_dir, getattr(cfg, "split_file", None)
        )
        samples = val_s if args.split == "val" else test_s

    ds = CaptchaDataset(samples, charset, cfg, training=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate_ctc)

    preds, labels = [], []
    for images, _, _, batch_labels in loader:
        logits = model(images.to(device))
        preds.extend(ctc_greedy_decode(logits, charset))
        labels.extend(batch_labels)

    m = compute_metrics(preds, labels)
    print(f"samples: {m['n']}  seq_acc: {m['seq_acc']:.4f}  char_acc: {m['char_acc']:.4f}")

    errors = [(p, t, path) for (path, _), p, t in zip(samples, preds, labels) if p != t]
    for p, t, path in errors[: args.show_errors]:
        print(f"  WRONG  gt={t!r:12} pred={p!r:12} {path.name}")
    if len(errors) > args.show_errors:
        print(f"  ... and {len(errors) - args.show_errors} more errors")
        
    if args.split == "test":
        error_list = [{"path": path.name, "gt": t, "pred": p, "confidence": 0.0} for p, t, path in errors]
        err_path = Path("errors.json")
        err_path.write_text(json.dumps(error_list, indent=2), encoding="utf-8")
        print(f"  Saved {len(error_list)} errors to {err_path.absolute()}")


if __name__ == "__main__":
    main()
