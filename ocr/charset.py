"""Charset auto-built from dataset filenames.

Filename convention: ``<label>_anything.<ext>`` — the ground truth is the part
of the stem before the first underscore. Index 0 is reserved for the CTC blank.
"""
from __future__ import annotations

import json
from pathlib import Path


def parse_label(path: str | Path) -> str:
    return Path(path).stem.split("_", 1)[0]


def scan_dataset(data_dir: str | Path, exts: tuple[str, ...]) -> list[tuple[Path, str]]:
    """Return [(path, label), ...] for all valid samples, sorted by filename."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    samples = []
    skipped = 0
    for p in sorted(data_dir.iterdir()):
        if p.suffix.lower() not in exts or not p.is_file():
            continue
        label = parse_label(p)
        if not label:
            skipped += 1
            continue
        samples.append((p, label))
    if skipped:
        print(f"[charset] skipped {skipped} files with empty label")
    if not samples:
        raise RuntimeError(f"No labeled images found in {data_dir}")
    return samples


class Charset:
    """Maps characters <-> integer ids. Id 0 is the CTC blank."""

    BLANK = 0

    def __init__(self, chars: str):
        self.chars = chars
        self._char_to_id = {c: i + 1 for i, c in enumerate(chars)}

    @classmethod
    def from_labels(cls, labels: list[str]) -> "Charset":
        return cls("".join(sorted(set("".join(labels)))))

    @property
    def num_classes(self) -> int:  # including blank
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        try:
            return [self._char_to_id[c] for c in text]
        except KeyError as e:
            raise KeyError(f"Character {e.args[0]!r} in label {text!r} not in charset") from e

    def decode_ids(self, ids) -> str:
        """CTC collapse: merge repeats, then drop blanks."""
        out, prev = [], self.BLANK
        for i in ids:
            i = int(i)
            if i != prev and i != self.BLANK:
                out.append(self.chars[i - 1])
            prev = i
        return "".join(out)


def save_meta(path: str | Path, charset: Charset, cfg) -> None:
    """Write everything a deployment (e.g. C++) needs to preprocess + decode."""
    if cfg.resize_mode == "stretch":
        pp = ("resize straight to (img_w, img_h), then x = pixel/127.5 - 1.0 "
              "(CHW float32); no padding")
    else:
        pp = ("resize height to img_h keeping aspect ratio (cap width at img_w), "
              "right-pad with black to img_w, then x = pixel/127.5 - 1.0 (CHW float32)")
    meta = {
        "charset": charset.chars,
        "blank_id": Charset.BLANK,
        "img_h": cfg.img_h,
        "img_w": cfg.img_w,
        "channels": cfg.channels,
        "resize_mode": cfg.resize_mode,
        "preprocess": pp,
        "decode": "argmax over classes per timestep, collapse repeats, drop blank_id",
    }
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
