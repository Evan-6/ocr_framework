import asyncio
from .db import SessionLocal, Job
from .runner import run_job_async

def _reconcile_orphans():
    """On startup, no job is actually running — mark stale 'running' jobs as failed."""
    db = SessionLocal()
    try:
        orphans = db.query(Job).filter(Job.status == "running").all()
        for job in orphans:
            job.status = "failed"
        if orphans:
            print(f"[queue] Reconciled {len(orphans)} orphaned 'running' job(s) -> failed")
        db.commit()
    finally:
        db.close()


async def job_queue_loop():
    """Background task that runs continuously, polling for pending jobs."""
    print("[queue] Worker loop started")
    _reconcile_orphans()
    while True:
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.status == "pending").order_by(Job.created_at.asc()).first()
            if job:
                job_id = job.id
                db.close()
                print(f"[queue] Starting job {job_id}")
                await run_job_async(job_id)
                print(f"[queue] Job {job_id} finished")
            else:
                db.close()
                await asyncio.sleep(2)
        except Exception as e:
            import traceback
            print(f"[queue] Error in loop: {type(e).__name__}: {e}")
            traceback.print_exc()
            db.close()
            await asyncio.sleep(2)
