import asyncio
import json
import os
import shutil
import subprocess
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from collections import Counter

from .db import Dataset, Job, get_db
from .events import sse_response
from .queue import job_queue_loop
from .runner import stop_job

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ocr.charset import Charset, scan_dataset
from ocr.splits import find_existing_split
from ocr.transforms import build_augment, preprocess
from ocr.trainer import load_checkpoint
from ocr.decoder import ctc_greedy_decode
from tools.gen_cpp_header import generate_header
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np

@asynccontextmanager
async def lifespan(app: FastAPI):
    # start queue loop
    task = asyncio.create_task(job_queue_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobCreate(BaseModel):
    name: str
    config_file: str
    config_overrides: str = "{}"

@app.post("/api/jobs")
def create_job(job_in: JobCreate, db: Session = Depends(get_db)):
    job = Job(
        name=job_in.name,
        config_file=job_in.config_file,
        config_overrides=job_in.config_overrides,
        status="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"id": job.id, "status": job.status}

@app.get("/api/jobs")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.created_at.desc()).all()
    return [{"id": j.id, "name": j.name, "status": j.status, "created_at": j.created_at, "metrics": j.metrics_dict} for j in jobs]

@app.get("/api/jobs/{job_id}")
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"id": job.id, "name": job.name, "status": job.status, "config": job.config_file, "overrides": job.overrides_dict, "metrics": job.metrics_dict}

@app.post("/api/jobs/{job_id}/stop")
async def stop_job_endpoint(job_id: int):
    await stop_job(job_id)
    return {"status": "stopping"}

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "running":
        raise HTTPException(status_code=400, detail="Stop the job before deleting it")
    out_dir = Path(f"runs/job_{job_id}")
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    db.delete(job)
    db.commit()
    return {"deleted": job_id}

@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: int):
    return sse_response(job_id)

@app.get("/api/jobs/{job_id}/metrics_history")
def get_job_metrics_history(job_id: int):
    out_dir = Path(f"runs/job_{job_id}")
    metrics_file = out_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return []
        
    history = []
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))
    return history

@app.get("/api/datasets")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).all()
    return datasets

@app.post("/api/datasets/sync")
def sync_datasets(db: Session = Depends(get_db)):
    synced = []
    for p in Path(".").iterdir():
        if p.is_dir() and p.name.startswith("data_"):
            if not db.query(Dataset).filter(Dataset.name == p.name).first():
                samples = scan_dataset(p, (".png", ".jpg", ".jpeg", ".bmp"))
                if samples:
                    charset = Charset.from_labels([lb for _, lb in samples])
                    max_len = max(len(lb) for _, lb in samples)
                    ds = Dataset(name=p.name, path=str(p).replace("\\", "/"), num_samples=len(samples), chars=charset.chars, max_length=max_len)
                    db.add(ds)
                    synced.append(p.name)
    db.commit()
    return {"synced": synced}

@app.delete("/api/datasets/{name}")
def delete_dataset(name: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.name == name).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    # Only remove files for uploaded datasets (those under datasets/), never a
    # project data_ folder that was merely synced/registered.
    p = Path(ds.path)
    removed_files = False
    try:
        if Path("datasets").resolve() in p.resolve().parents and p.exists():
            shutil.rmtree(p, ignore_errors=True)
            removed_files = True
    except Exception:
        pass
    db.delete(ds)
    db.commit()
    return {"deleted": name, "files_removed": removed_files}

@app.get("/api/datasets/{name}/samples")
def get_dataset_samples(name: str, page: int = 1, page_size: int = 50, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.name == name).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    
    samples = scan_dataset(Path(ds.path), (".png", ".jpg", ".jpeg", ".bmp"))
    total = len(samples)
    start = (page - 1) * page_size
    end = start + page_size
    
    split_path = find_existing_split(name)
    split_map = {}
    if split_path and split_path.exists():
        sp = json.loads(split_path.read_text(encoding="utf-8"))
        for s in sp.get("train", []): split_map[s] = "train"
        for s in sp.get("val", []): split_map[s] = "val"
        for s in sp.get("test", []): split_map[s] = "test"
        
    res = []
    for fpath, label in samples[start:end]:
        fname = Path(fpath).name
        res.append({
            "filename": fname,
            "label": label,
            "split": split_map.get(fname, "unknown")
        })
    return {"total": total, "samples": res}

@app.get("/api/datasets/{name}/stats")
def get_dataset_stats(name: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.name == name).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
        
    samples = scan_dataset(Path(ds.path), (".png", ".jpg", ".jpeg", ".bmp"))
    
    lengths = Counter(len(lb) for _, lb in samples)
    chars = Counter()
    for _, lb in samples:
        chars.update(lb)
        
    split_path = find_existing_split(name)
    splits_count = {"train": 0, "val": 0, "test": 0, "unknown": len(samples)}
    if split_path and split_path.exists():
        sp = json.loads(split_path.read_text(encoding="utf-8"))
        splits_count["train"] = len(sp.get("train", []))
        splits_count["val"] = len(sp.get("val", []))
        splits_count["test"] = len(sp.get("test", []))
        splits_count["unknown"] = len(samples) - (splits_count["train"] + splits_count["val"] + splits_count["test"])
        
    return {
        "lengths": dict(lengths),
        "chars": dict(chars.most_common(20)),
        "splits": splits_count
    }

@app.get("/api/datasets/{name}/images/{filename}")
def get_dataset_image(name: str, filename: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.name == name).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    filepath = Path(ds.path) / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(filepath)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _valid_image_basename(fn: str) -> bool:
    """Filename must have a label (part before '_') and an image extension."""
    base = os.path.basename(fn)  # zip-slip: flatten away any path
    return bool(base) and "_" in base and os.path.splitext(base)[1].lower() in IMG_EXTS


@app.post("/api/datasets/upload")
async def upload_dataset(name: str = Form(...), files: List[UploadFile] = File(...),
                         db: Session = Depends(get_db)):
    """Accept loose images and/or a .zip. Labels come from filenames (<label>_*.ext)."""
    if db.query(Dataset).filter(Dataset.name == name).first():
        raise HTTPException(status_code=400, detail="Dataset name already exists")

    dataset_dir = Path("datasets") / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0

    def save_from(src, filename: str):
        nonlocal saved, skipped
        if not _valid_image_basename(filename):
            skipped += 1
            return
        dest = dataset_dir / os.path.basename(filename)
        if dest.exists():   # duplicate policy: skip
            skipped += 1
            return
        with open(dest, "wb") as target:
            shutil.copyfileobj(src, target)
        saved += 1

    try:
        for uf in files:
            fn = uf.filename or ""
            if fn.lower().endswith(".zip"):
                zip_path = dataset_dir / "_upload.zip"
                with open(zip_path, "wb") as buffer:
                    shutil.copyfileobj(uf.file, buffer)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    for info in zip_ref.infolist():
                        if info.is_dir():
                            continue
                        if not _valid_image_basename(info.filename):
                            skipped += 1
                            continue
                        with zip_ref.open(info) as source:
                            save_from(source, info.filename)
                zip_path.unlink()
            else:
                save_from(uf.file, fn)
    except zipfile.BadZipFile:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

    if saved == 0:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"No valid labeled images found. Skipped {skipped}.")

    # Scan dataset to build metadata
    samples = scan_dataset(dataset_dir, IMG_EXTS)
    charset = Charset.from_labels([lb for _, lb in samples])
    max_len = max(len(lb) for _, lb in samples) if samples else 0
    
    ds = Dataset(
        name=name,
        path=str(dataset_dir).replace("\\", "/"),
        num_samples=len(samples),
        chars=charset.chars,
        max_length=max_len
    )
    db.add(ds)
    db.commit()
    
    return {"name": name, "saved": saved, "skipped": skipped, "chars": charset.chars, "max_length": max_len}

@app.post("/api/jobs/{job_id}/export")
def export_job(job_id: int, db: Session = Depends(get_db)):
    """Export best.pt -> ONNX (with parity check) and generate the C++ config header."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    run_dir = Path(f"runs/job_{job_id}")
    best = run_dir / "best.pt"
    if not best.exists():
        raise HTTPException(status_code=400, detail="No checkpoint (best.pt) for this job yet")

    onnx_path = run_dir / "model.onnx"
    proc = subprocess.run(
        [sys.executable, "export_onnx.py", "--ckpt", str(best), "--out", str(onnx_path)],
        capture_output=True, text=True, cwd=os.getcwd(),
    )
    if not onnx_path.exists():
        raise HTTPException(status_code=500,
                            detail=f"Export failed:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")

    parity_ok = proc.returncode == 0  # export_onnx exits non-zero if parity fails
    header_written = False
    meta_path = run_dir / "model_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        (run_dir / "ocr_model_config.h").write_text(generate_header(meta), encoding="utf-8")
        header_written = True

    size_mb = round(onnx_path.stat().st_size / 1048576, 2)
    return {"parity_ok": parity_ok, "onnx_size_mb": size_mb,
            "header": header_written, "log": proc.stdout[-1500:]}


ARTIFACTS = {"onnx": "model.onnx", "meta": "model_meta.json",
             "header": "ocr_model_config.h", "best": "best.pt"}

@app.get("/api/jobs/{job_id}/download/{artifact}")
def download_artifact(job_id: int, artifact: str):
    if artifact not in ARTIFACTS:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    path = Path(f"runs/job_{job_id}") / ARTIFACTS[artifact]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found — export first")
    return FileResponse(path, filename=path.name)

@app.get("/api/jobs/{job_id}/errors")
def get_job_errors(job_id: int):
    out_dir = Path(f"runs/job_{job_id}")
    err_file = out_dir / "errors.json"
    if not err_file.exists():
        return []
    return json.loads(err_file.read_text(encoding="utf-8"))

class DummyCfg:
    pass

@app.post("/api/aug/preview")
def aug_preview(
    file: UploadFile = File(...), 
    aug_rotate: float = Form(0.0),
    aug_translate: float = Form(0.0),
    aug_scale: float = Form(0.0),
    aug_shear: float = Form(0.0),
    aug_photometric: bool = Form(False),
    channels: int = Form(1)
):
    cfg = DummyCfg()
    cfg.aug_rotate = aug_rotate
    cfg.aug_translate = aug_translate
    cfg.aug_scale = aug_scale
    cfg.aug_shear = aug_shear
    cfg.aug_photometric = aug_photometric
    cfg.channels = channels
    
    aug_fn = build_augment(cfg)
    img = Image.open(file.file).convert("RGB")
    if not aug_fn:
        buf = BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"images": [f"data:image/jpeg;base64,{b64}"]}
        
    res = []
    for _ in range(4):
        aug_img = aug_fn(img)
        buf = BytesIO()
        aug_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        res.append(f"data:image/jpeg;base64,{b64}")
    return {"images": res}

class ModelCache:
    job_id = None
    model = None
    charset = None
    cfg = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

@app.post("/api/jobs/{job_id}/infer")
def run_infer(job_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job or job.status != "done":
        raise HTTPException(400, "Job not finished")
        
    ckpt_path = Path(f"runs/job_{job_id}/best.pt")
    if not ckpt_path.exists():
        raise HTTPException(400, "Checkpoint not found")
        
    if ModelCache.job_id != job_id:
        ModelCache.model, ModelCache.charset, ModelCache.cfg = load_checkpoint(str(ckpt_path), ModelCache.device)
        ModelCache.job_id = job_id
        
    img = Image.open(file.file)
    batch = np.stack([preprocess(img, ModelCache.cfg.img_h, ModelCache.cfg.img_w, ModelCache.cfg.channels, ModelCache.cfg.resize_mode)])
    
    with torch.no_grad():
        preds = ModelCache.model(torch.from_numpy(batch).to(ModelCache.device)).cpu().numpy()
        
    text = ctc_greedy_decode(preds, ModelCache.charset)[0]
    return {"text": text}
