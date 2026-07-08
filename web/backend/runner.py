import asyncio
import os
import subprocess
import sys
from pathlib import Path

from .db import Job, SessionLocal

# Global dictionary to store running processes for cancellation
RUNNING_PROCESSES = {}

async def run_job_async(job_id: int):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        return

    out_dir = Path(f"runs/job_{job_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    job.status = "running"
    job.out_dir = str(out_dir)
    db.commit()

    import yaml
    from ocr.config import Config
    
    # Generate job-specific config
    try:
        base_cfg = job.config_file if job.config_file else None
        cfg = Config.load(base_cfg)
        
        for k, v in job.overrides_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                
        run_cfg_path = out_dir / "config.yaml"
        with open(run_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_dict(), f)
            
        # sys.executable is the venv python running this server — an absolute path.
        # Do NOT use a relative ".venv/Scripts/python.exe": Windows CreateProcess can't
        # resolve a relative exe with forward slashes (WinError 2).
        cmd = [sys.executable, "train.py", "--config", str(run_cfg_path), "--out-dir", str(out_dir)]
    except Exception as e:
        job.status = "failed"
        db.commit()
        db.close()
        print(f"[runner] Config generation failed: {e}")
        return

    log_file = out_dir / "train.log"
    # NOTE: do NOT use asyncio.create_subprocess_exec here. On Windows it needs a
    # ProactorEventLoop, but uvicorn runs a SelectorEventLoop, which raises a bare
    # NotImplementedError (empty message). Run a blocking Popen in a thread instead —
    # loop-agnostic, and fine for a single-worker serialized queue.
    loop = asyncio.get_running_loop()
    log_handle = open(log_file, "w", encoding="utf-8")
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT,
            cwd=os.getcwd(), env=env,
        )
        RUNNING_PROCESSES[job_id] = process
        job.pid = process.pid
        db.commit()
        returncode = await loop.run_in_executor(None, process.wait)
    except Exception as e:
        job.status = "failed"
        db.commit()
        db.close()
        print(f"[runner] Failed to launch training for job {job_id}: {type(e).__name__}: {e}")
        return
    finally:
        log_handle.close()
        RUNNING_PROCESSES.pop(job_id, None)

    # Re-fetch job to avoid stale state if it was cancelled by user
    db.refresh(job)

    if job.status == "running":  # not explicitly cancelled
        job.status = "done" if returncode == 0 else "failed"

    # Read final metrics if available
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        try:
            job.metrics = metrics_path.read_text(encoding="utf-8")
        except Exception:
            pass

    db.commit()
    db.close()

async def stop_job(job_id: int):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if job and job.status == "running":
        job.status = "stopped"
        db.commit()
        
        process = RUNNING_PROCESSES.get(job_id)
        if process:
            try:
                if os.name == 'nt':
                    # On Windows, use taskkill to kill the tree
                    import subprocess
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
                else:
                    process.terminate()
            except Exception as e:
                print(f"[runner] Error stopping job {job_id}: {e}")
    db.close()
