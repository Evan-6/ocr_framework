# Captcha OCR Framework (CRNN + CTC)

A general OCR captcha recognizer for **fixed- or variable-length** codes, built in
PyTorch and designed to export cleanly to **ONNX for C++ / ONNX Runtime** deployment
on Windows.

Dataset = one folder of images named `<label>_anything.ext`; the ground truth is the
part of the stem before the first underscore (`ABCD_001.png -> ABCD`,
`123456_test.jpg -> 123456`). The charset is built automatically from the filenames.

---

## 1. Why CRNN + CTC (model analysis)

The requirements were: variable length first (fixed length too), good on small
datasets, fast inference, easy to train, stable ONNX export, ONNX Runtime C++
compatibility (no exotic operators), and easy to extend. Here is how the candidates
compare against exactly those constraints:

| Model | Variable len | Small data | Speed | ONNX / ORT C++ | Train ease |
|-------|:---:|:---:|:---:|:---:|:---:|
| **CRNN + CTC** | ✅ native | ✅ excellent | ✅ fast | ✅ Conv/LSTM/Linear only | ✅ easy |
| SVTR + CTC | ✅ | ⚠️ needs more data | ✅ | ⚠️ some reshape/attn quirks | ⚠️ |
| PARSeq | ✅ | ❌ data hungry | ⚠️ AR decode | ⚠️ autoregressive loop | ❌ |
| ABINet | ✅ | ❌ needs LM pretrain | ❌ | ❌ complex | ❌ |
| SATRN | ✅ | ❌ | ❌ | ⚠️ transformer ops | ❌ |

**CRNN + CTC wins on every constraint that matters here.** CTC handles variable
length natively (no per-length heads, no max-length assumption baked into the graph),
converges on hundreds of images, and the whole network is just
`Conv2d / BatchNorm / ReLU / MaxPool / LSTM / Linear` — all first-class, well-optimized
ONNX Runtime operators with no autoregressive decode loop to unroll. The
attention/transformer models (PARSeq, ABINet, SATRN) reach slightly higher accuracy on
large public benchmarks but are data-hungry, slower, and harder to export — the wrong
trade for small captcha datasets that must ship to C++.

The backbone/neck/head are separated (`ocr/model.py`), so a stronger backbone (e.g.
SVTR blocks) can be dropped in later behind the same CTC head and training pipeline.

Verified on the two sample datasets:

| Dataset | Images | Size | Length | Charset | Kind | Val seq-acc |
|---------|-------:|------|--------|---------|------|:-----------:|
| data_2 | 1960 | 196×44 | **variable 4–7** | 29 chars | text captcha | **100%** @ epoch 36 |
| data_1 | 257 | 500×150 | fixed 4 | `d l r u` | directional-arrow captcha | **100%** @ epoch 38 (RGB) |

> **data_1 is not a text captcha.** Each label char is the *direction* of one of four
> arrows in a horizontal band over cluttered game screenshots (`d`own/`l`eft/`r`ight/`u`p,
> e.g. `dddr` = down·down·down·right). The framework handles it the same way — 4 symbols
> read left-to-right — but two dataset-specific settings were decisive:
>
> - **Color (`channels: 3`)** — the arrows are saturated colored shapes that nearly vanish
>   in grayscale against the desaturated clutter. Switching to RGB took val seq-acc from
>   **83% → 100%**, and it converged far faster (100% by epoch 38 vs 83% only by epoch 175).
> - **No rotation/shear** — those operations change an arrow's direction and therefore its
>   label (see §2).
>
> Grayscale is still the right default for ordinary text captchas (faster, and data_2 hits
> 100% with 1 channel); color is opt-in per dataset via config.

## 2. Preprocessing

The two given sizes (196×44 ≈ 4.45:1 and 500×150 ≈ 3.33:1) have different aspect
ratios, so a naive stretch to a common box would distort glyphs differently per source.
Instead:

1. Convert to grayscale (captcha color is rarely informative; 1 channel = faster).
2. **Resize height to `img_h` (48) keeping aspect ratio**, capping width at `img_w`.
3. **Right-pad with black** to a fixed `img_w` (224).
4. Normalize `x / 127.5 - 1.0` → range `[-1, 1]`, layout CHW float32.

Fixed height keeps the CNN receptive field consistent; aspect-preserving width keeps
characters unsquashed; padding gives a fixed tensor for batching and a fixed ONNX shape.
The CNN downsamples width ×4, giving **56 CTC timesteps** — comfortably above
`2·max_len+1` (needed for CTC) for labels up to length 7 (and up to ~27 chars).

`img_h` must be divisible by 16 and `img_w` by 4 (enforced in `Config`). This exact
recipe is mirrored in `cpp_example/main.cpp` and described in `model_meta.json`.

**Augmentation is configurable and orientation-aware.** Geometric jitter
(`aug_rotate`, `aug_translate`, `aug_scale`, `aug_shear`) and photometric jitter
(`aug_photometric`: brightness/contrast/blur/noise) are toggled independently. For
ordinary text captchas the defaults (small rotation + shear) help generalization. For
**orientation-sensitive** captchas like data_1's arrows, `configs/data_1.yaml` sets
`aug_rotate: 0` and `aug_shear: 0` — rotating a "down" arrow toward "down-right" would
silently corrupt its label — while keeping photometric noise on to survive the varied
backgrounds. Inference/export use no augmentation.

## 3. Architecture

```
input (B, 1, 48, 224)
  └─ CNN backbone   (VGG-style, 7 conv blocks) → (B, 512, 1, 56)
  └─ squeeze+permute                            → (B, 56, 512)
  └─ BiLSTM neck    (2× bidirectional, 256 hid) → (B, 56, 512)
  └─ Linear head                                → (B, 56, num_classes) logits
  └─ CTC (train) / greedy decode (infer)
```

Index 0 is the CTC blank. Decoding = argmax per timestep → collapse repeats → drop
blanks. ~8.7M parameters at default width.

> Note: the neck uses two stacked single-layer `nn.LSTM`s with explicit `Dropout`
> rather than one `nn.LSTM(num_layers=2, dropout=…)`. cuDNN's fused multi-layer dropout
> can crash at process teardown on some Windows/CUDA builds, and separate layers also
> export to cleaner ONNX.

### Model size (lightweight variants)

The default net is ~8.7M params ≈ **35 MB** float32 — heavy to embed. Two knobs scale it
down with no architecture change: `stage_channels` (the four backbone widths) and
`hidden` (LSTM size). Ready-made configs for data_1 (arrow captcha):

| Variant | `stage_channels` | `hidden` | Params | ONNX size | data_1 val seq-acc | all-257 |
|---------|------------------|:--------:|-------:|----------:|:------------------:|:-------:|
| full   | 64,128,256,512 | 256 | 8.71M | 33.2 MB | 100% | 100% |
| **~10 MB** | 48,96,160,224 | 160 | 2.50M | 9.55 MB | 100% | 100% |
| **~5 MB**  | 32,64,96,160  | 112 | 1.18M | 4.51 MB | 100% | 100% |

Both lightweight variants reach the **same 100%** on data_1 — the arrow captcha doesn't
need the full capacity once it's in color. They just take a few more epochs to break the
CTC plateau (100% val by ~epoch 58–67 vs 38 for the full model). Train them with
`configs/data_1_5mb.yaml` / `configs/data_1_10mb.yaml`.

> For a *perfect fit on every training image* (not just held-out val), let the cosine LR
> decay all the way to ~0 — i.e. train the full `epochs` rather than early-stopping mid-
> schedule. `--early-stop-patience 999` disables early stop when you want that last bit.

To size a variant for a harder dataset, pick `stage_channels`/`hidden` and check params
up front:

```python
from ocr.config import Config; from ocr.model import build_model
cfg = Config.load("configs/data_1_5mb.yaml")
m = build_model(cfg, num_classes=5)
p = sum(x.numel() for x in m.parameters()); print(f"{p/1e6:.2f}M params, ~{p*4/1e6:.1f} MB")
```

## 4. Setup

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## 5. Usage

```powershell
# Train (charset auto-built from filenames)
.venv\Scripts\python.exe train.py --config configs\data_2.yaml
.venv\Scripts\python.exe train.py --data-dir data_1 --out-dir runs\data_1 --epochs 300

# Evaluate (re-derives the training-time val split)
.venv\Scripts\python.exe evaluate.py --ckpt runs\data_2\best.pt --show-errors 20

# Inference — PyTorch
.venv\Scripts\python.exe infer.py --ckpt runs\data_2\best.pt --input some_folder

# Export to ONNX (runs an automatic PyTorch-vs-ORT parity check)
.venv\Scripts\python.exe export_onnx.py --ckpt runs\data_2\best.pt

# Inference — ONNX Runtime (same code path C++ will use)
.venv\Scripts\python.exe infer.py --onnx runs\data_2\model.onnx --meta runs\data_2\model_meta.json --input some_folder
```

Each run directory contains: `best.pt` / `last.pt` (self-contained checkpoints with
charset + config), `model.onnx`, `model_meta.json` (charset + preprocessing + decode
spec for C++), and `config.json`.

## 6. ONNX / C++ deployment

- **Input** `input`: float32 `(batch, channels, img_h, img_w)`, dynamic batch axis.
- **Output** `logits`: float32 `(batch, T, num_classes)`; argmax per timestep + CTC
  collapse (drop repeats, drop blank id 0) in C++.
- Opset 17, no custom ops — runs on stock ONNX Runtime CPU / CUDA / DirectML EP.
- All preprocessing/decode constants are in `model_meta.json` (`charset`, `img_h`,
  `img_w`, `channels`, `resize_mode`, `preprocess`, `decode`).

`cpp_example/` targets the **data_1 5 MB** model with ONNX Runtime + DirectML (CPU
fallback):
- `ocr.cpp` — loads the model (from a Windows resource), preprocesses, runs, argmaxes.
  Because the model was trained with `resize_mode: stretch`, preprocessing is a single
  `cv::resize` to 224×64 + RGB + `x/127.5 − 1.0` (no blur/crop/padding).
- `ocr_label_decoder.h` — CTC collapse; `LabelMap` maps class ids to chars and **must
  match `model_meta.json`'s `charset`** (blank + `dlru` → `{"","d","l","r","u"}`).
- Input node `input` `(1,3,64,224)`, output node `logits` `(1,T,5)`.

## 7. Project layout

```
ocr/
  config.py      dataclass config, YAML + CLI overrides, validation
  charset.py     filename→label parsing, auto charset, meta export
  transforms.py  resize+pad preprocessing, train-time augmentation
  dataset.py     Dataset + CTC collate + deterministic split
  model.py       CRNN (backbone / neck / head, swappable)
  decoder.py     CTC greedy decode (torch or numpy)
  metrics.py     sequence acc + char acc (edit distance)
  trainer.py     train loop: CTC, AMP, warmup+cosine, early stop, checkpoint
train.py  evaluate.py  infer.py  export_onnx.py
configs/  data_1.yaml  data_2.yaml
cpp_example/main.cpp
```

## 8. Extending

- **Stronger backbone**: swap `CNNBackbone` in `ocr/model.py` (keep the `(B, T, C)`
  output contract); training/export are untouched.
- **Beam search / language model**: add a decoder in `ocr/decoder.py`; the ONNX graph
  still outputs raw logits, so decode strategy stays outside the model.
- **New dataset**: just point `--data-dir` at a folder — charset and lengths adapt
  automatically.
- **Config**: add a field to `Config`; it is immediately available via YAML and (if you
  add the arg) the CLI.
```
