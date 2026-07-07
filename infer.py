"""Run inference on an image or a folder, with PyTorch or ONNX Runtime.

    python infer.py --ckpt runs/data_2/best.pt --input some.png
    python infer.py --ckpt runs/data_2/best.pt --input folder/
    python infer.py --onnx runs/data_2/model.onnx --meta runs/data_2/model_meta.json --input folder/
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ocr.charset import Charset, parse_label
from ocr.decoder import ctc_greedy_decode
from ocr.transforms import preprocess


def collect_images(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--ckpt", help="PyTorch checkpoint (best.pt)")
    ap.add_argument("--onnx", help="ONNX model (uses onnxruntime instead of PyTorch)")
    ap.add_argument("--meta", help="model_meta.json (required with --onnx)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if args.onnx:
        if not args.meta:
            ap.error("--meta is required with --onnx")
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        charset = Charset(meta["charset"])
        img_h, img_w, channels = meta["img_h"], meta["img_w"], meta["channels"]
        resize_mode = meta.get("resize_mode", "pad")
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        def run(batch: np.ndarray) -> np.ndarray:
            return sess.run(None, {input_name: batch})[0]
    else:
        if not args.ckpt:
            ap.error("provide --ckpt or --onnx")
        import torch
        from ocr.trainer import load_checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, charset, cfg = load_checkpoint(args.ckpt, device)
        img_h, img_w, channels = cfg.img_h, cfg.img_w, cfg.channels
        resize_mode = cfg.resize_mode

        @torch.no_grad()
        def run(batch: np.ndarray) -> np.ndarray:
            return model(torch.from_numpy(batch).to(device)).cpu().numpy()

    files = collect_images(args.input)
    if not files:
        raise SystemExit(f"No images found under {args.input}")

    n_correct, n_labeled = 0, 0
    t0 = time.time()
    for i in range(0, len(files), args.batch_size):
        chunk = files[i : i + args.batch_size]
        batch = np.stack([preprocess(Image.open(f), img_h, img_w, channels, resize_mode)
                          for f in chunk])
        preds = ctc_greedy_decode(run(batch), charset)
        for f, pred in zip(chunk, preds):
            gt = parse_label(f)
            mark = ""
            if gt and set(gt) <= set(charset.chars):
                n_labeled += 1
                n_correct += pred == gt
                mark = "  OK" if pred == gt else f"  WRONG (gt={gt})"
            print(f"{f.name:40} -> {pred}{mark}")
    dt = time.time() - t0
    print(f"\n{len(files)} images in {dt:.2f}s ({dt / len(files) * 1000:.1f} ms/img)")
    if n_labeled:
        print(f"accuracy on labeled files: {n_correct}/{n_labeled} = {n_correct / n_labeled:.4f}")


if __name__ == "__main__":
    main()
