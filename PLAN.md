# OCR Framework — Development Plan

Status: **plan confirmed, not yet implemented.** Frontend stack decided: **React + Vite SPA**.
Backend: **FastAPI + SQLite + single-GPU job worker**.

A "task" = a training job (a config + its run + artifacts + metrics).

---

## Phase 1 — Data & evaluation correctness (foundation, do first)

### 1.1 Three-way split (train / val / test)
- `split_samples(samples, val_ratio, test_ratio, seed)` → `(train, val, test)`.
- **Test set held out from training AND model selection.** Val drives early-stopping /
  `best.pt`; test is evaluated once for the final unbiased number.
- **Persist the split per dataset** → `splits/<dataset>__seed<seed>__v<val>__t<test>.json`
  (exact filenames per set). Every job on that dataset reuses the same test set, so runs
  are comparable in the web UI. This is "split the test set at task creation."
- Optional **stratification by label length** (matters for data_2's 4–7 mix).
- Config additions: `test_ratio: float`, `split_file: str|null` (explicit override).

### 1.2 Charset consistency
- Build charset from **train ∪ val** only.
- At test time, **warn/flag** labels containing a char unseen in train∪val (don't silently
  extend the vocabulary from test).

### 1.3 Reconcile early-stopping with intent
- Track **val loss** in `validate()` (currently only accuracy is computed — the config
  comment about "Val Loss" does not match the code today).
- `select_metric: val_loss | seq_acc | char_acc` + `select_mode: min | max`. `best.pt` and
  early-stop both key off this metric.

### 1.4 Richer metrics + artifacts per run
- Per-position accuracy, **per-class confusion** (d/l/r/u arrows), mean confidence, val-loss
  curve.
- Write `metrics.json` (final train/val/**test** numbers), `metrics.jsonl` (per-epoch, for
  live charts), `errors.json` (misclassified test files: path, gt, pred, confidence).

### 1.5 Reproducibility metadata
- Each run records: full config, seed, git commit, environment (torch/cuda/ort versions),
  dataset hash, split hash.

**Files touched:** `ocr/dataset.py`, `ocr/trainer.py`, `ocr/metrics.py`, `ocr/config.py`,
`evaluate.py` (add `--split test`), new `ocr/splits.py`.

---

## Phase 2 — Web app: task / experiment manager

### 2.1 Architecture
```
web/
  backend/            FastAPI app
    api.py            REST endpoints (jobs, datasets, models, inference)
    db.py             SQLite (jobs, metrics, artifacts, datasets, splits)
    queue.py          single-GPU worker: queued -> running -> done/failed/stopped
    runner.py         launches train.py as subprocess, tails metrics.jsonl
    events.py         SSE/WebSocket push of live metrics + log lines
  frontend/           React + Vite SPA
    (jobs list, create-job form, live run view, compare, dataset browser,
     aug preview, error gallery, inference playground, model registry)
```
- **Single GPU (RTX 2080) → one worker, jobs serialize.** Queue supports cancel/stop.
- Training stays a subprocess of the existing `train.py` (no rewrite); the trainer just
  needs to emit `metrics.jsonl` and honor a stop signal.
- Live updates via **SSE** (simpler than WebSocket for one-way metric/log streaming).

### 2.2 Feature checklist
- **Create job**: form for every config field, validation, inline docs, **presets**
  ("arrow 5MB", "text grayscale"), clone-from-existing.
- **Queue & control**: start / stop / cancel / re-run.
- **Live run view**: loss & acc curves, log tail, ETA, GPU util/mem.
- **Compare runs**: overlay curves + final metrics table on the shared test set.
- **Dataset browser**: thumbnails, charset preview, length histogram, class balance,
  filename-format validation, **train/val/test split viewer**.
- **Augmentation preview**: render N augmented samples for the current aug config (high
  value while tuning aug).
- **Error gallery**: misclassified **test** images, gt vs pred vs confidence.
- **Inference playground**: drag-drop image → prediction + confidence (PyTorch or ONNX).

### 2.3 REST surface (sketch)
```
POST /jobs            create+queue    GET /jobs            list
GET  /jobs/{id}       detail+metrics  POST /jobs/{id}/stop
GET  /jobs/{id}/events  SSE stream    GET  /jobs/{id}/artifacts
GET  /datasets        list+stats      GET  /datasets/{d}/split
POST /aug/preview     augmented imgs  POST /infer  (image -> pred)
POST /models/{id}/publish   -> Phase 3
```

---

## Phase 3 — Deployment automation (closes the drift bug we hit)

- One-click **Publish**: export ONNX → PyTorch/ORT parity check → size report →
  **auto-generate the C++ header** (`charset`→`LabelMap`, `img_h/img_w/channels`,
  `resize_mode`) from `model_meta.json`.
  - This eliminates the class of bug we fixed by hand (the `l,r,u,d` vs `d,l,r,u`
    `LabelMap` mismatch). The header becomes generated, never hand-edited.
- **Model registry**: versioned models + their **test** metrics; downloadable bundle
  (`model.onnx` + `model_meta.json` + generated header).

**New:** `tools/gen_cpp_header.py`, wired into `export_onnx.py --emit-cpp` and the publish
endpoint.

---

## Phase 4 — Engineering quality

- **Tests**: split determinism, CTC decode, preprocess↔ONNX parity, charset round-trip,
  fast smoke-train. CI-friendly.
- **Param sweep**: enqueue a grid of configs from the UI (e.g. sweep `aug_scale`, `hidden`).
- **Structured logging** + optional TensorBoard.
- **Packaging**: dockerize API + frontend; GPU worker stays native on Windows.

---

## Recommended execution order
1. **Phase 1** (correct evaluation underpins everything the UI compares).
2. **Thin vertical slice of Phase 2**: create-job → queue → live status → run list.
3. Broaden Phase 2 features (compare, dataset browser, aug preview, error gallery, playground).
4. **Phase 3** deployment automation.
5. **Phase 4** hardening.

## Open decisions
- Test ratio default (suggest `test_ratio: 0.1`, `val_ratio: 0.1`).
- Default `select_metric` (suggest `val_loss` per your early-stop intent; note it can diverge
  from seq-acc on tiny sets — worth exposing both curves).
- Whether the web app should also **launch inference/serving** for production, or stay a
  training manager only.
