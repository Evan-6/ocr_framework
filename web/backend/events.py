import asyncio
from pathlib import Path
from fastapi.responses import StreamingResponse

async def tail_events(job_id: int):
    out_dir = Path(f"runs/job_{job_id}")
    metrics_file = out_dir / "metrics.jsonl"
    log_file = out_dir / "train.log"
    
    metrics_pos = 0
    log_pos = 0
    
    try:
        while True:
            updated = False
            
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    f.seek(metrics_pos)
                    lines = f.readlines()
                    if lines:
                        metrics_pos = f.tell()
                        for line in lines:
                            if line.strip():
                                yield f"event: metrics\ndata: {line.strip()}\n\n"
                        updated = True
                        
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(log_pos)
                    lines = f.readlines()
                    if lines:
                        log_pos = f.tell()
                        for line in lines:
                            if line.strip():
                                # simple JSON encoding to escape quotes for the data field
                                import json
                                safe_line = json.dumps(line.strip())
                                yield f"event: log\ndata: {safe_line}\n\n"
                        updated = True
                        
            if not updated:
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

def sse_response(job_id: int):
    return StreamingResponse(tail_events(job_id), media_type="text/event-stream")
