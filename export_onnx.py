"""Export a checkpoint to ONNX and verify parity with ONNX Runtime.

    python export_onnx.py --ckpt runs/data_2/best.pt --out runs/data_2/model.onnx

Input : "input"  float32 (batch, channels, img_h, img_w), dynamic batch
Output: "logits" float32 (batch, timesteps, num_classes) — argmax + CTC collapse in C++.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from ocr.charset import scan_dataset
from ocr.decoder import ctc_greedy_decode
from ocr.trainer import load_checkpoint
from ocr.transforms import preprocess


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", help="output .onnx path (default: <ckpt dir>/model.onnx)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model, charset, cfg = load_checkpoint(args.ckpt, "cpu")
    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "model.onnx"

    dummy = torch.randn(1, cfg.channels, cfg.img_h, cfg.img_w)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, dynamo=False,
    )
    print(f"[export] {out_path} (opset {args.opset})")

    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))
    print("[check] onnx.checker passed")

    # --- parity check: PyTorch vs ONNX Runtime on real images if available ---
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    try:
        from PIL import Image
        samples = scan_dataset(cfg.data_dir, cfg.exts)[:8]
        batch = np.stack([preprocess(Image.open(p), cfg.img_h, cfg.img_w, cfg.channels,
                                     cfg.resize_mode) for p, _ in samples])
    except (FileNotFoundError, RuntimeError):
        batch = np.random.randn(4, cfg.channels, cfg.img_h, cfg.img_w).astype(np.float32)

    with torch.no_grad():
        ref = model(torch.from_numpy(batch)).numpy()
    out = sess.run(None, {"input": batch})[0]
    max_diff = float(np.abs(ref - out).max())
    pt_dec = ctc_greedy_decode(ref, charset)
    ort_dec = ctc_greedy_decode(out, charset)
    print(f"[verify] batch={batch.shape[0]}, max |pytorch - ort| = {max_diff:.2e}")
    print(f"[verify] decoded parity: {'OK' if pt_dec == ort_dec else 'MISMATCH'} {ort_dec}")
    if max_diff > 1e-3 or pt_dec != ort_dec:
        raise SystemExit("Parity check FAILED")
    print("[done] ONNX model verified against PyTorch")


if __name__ == "__main__":
    main()
