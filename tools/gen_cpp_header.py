"""Generate a C++ header of model constants from model_meta.json.

This is the anti-drift tool: the C++ ``LabelMap`` (id -> char), input size,
channels, resize mode and IO names are all derived from the trained model's
metadata, so the deployment can never silently disagree with the model (the
class of bug where charset order was hand-mistyped as l,r,u,d vs d,l,r,u).

    python tools/gen_cpp_header.py --meta runs/job_5/model_meta.json --out runs/job_5/ocr_model_config.h
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cpp_str(s: str) -> str:
    """Escape a Python string for a C++ double-quoted string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def generate_header(meta: dict, input_name: str = "input", output_name: str = "logits") -> str:
    charset = meta["charset"]
    # Index 0 is the CTC blank; ids 1.. map to charset chars in order.
    entries = ['""'] + [f'"{_cpp_str(c)}"' for c in charset]
    label_map = ", ".join(entries)
    return f"""// Auto-generated from model_meta.json by tools/gen_cpp_header.py — DO NOT EDIT.
#pragma once

#include <cstdint>

namespace vision::ocr {{

// ---- preprocessing / input ----
constexpr int kImgH = {meta["img_h"]};
constexpr int kImgW = {meta["img_w"]};
constexpr int kChannels = {meta["channels"]};
constexpr const char* kResizeMode = "{meta.get("resize_mode", "pad")}";  // "pad" | "stretch"
constexpr float kNormScale = 127.5f;   // x = pixel / kNormScale - 1.0
constexpr float kNormBias = 1.0f;

// ---- ONNX IO node names ----
constexpr const char* kInputName = "{input_name}";
constexpr const char* kOutputName = "{output_name}";

// ---- CTC label map (index 0 = blank; 1.. = charset order) ----
inline constexpr const char* LabelMap[] = {{ {label_map} }};
constexpr int kLabelCount = {len(charset) + 1};

}}  // namespace vision::ocr
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="path to model_meta.json")
    ap.add_argument("--out", help="output .h path (default: <meta dir>/ocr_model_config.h)")
    ap.add_argument("--input-name", default="input")
    ap.add_argument("--output-name", default="logits")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else Path(args.meta).parent / "ocr_model_config.h"
    out.write_text(generate_header(meta, args.input_name, args.output_name), encoding="utf-8")
    print(f"[gen_cpp_header] wrote {out} (charset={meta['charset']!r})")


if __name__ == "__main__":
    main()
