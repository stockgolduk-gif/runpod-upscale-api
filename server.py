import os
import uuid
import json
import time
import threading
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

# ----------------------------
# Config
# ----------------------------
BASE_DIR = "/workspace"
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
REGISTRY_PATH = os.path.join(JOBS_DIR, "registry.json")

os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory registry (mirrored to disk)
JOB_REGISTRY: Dict[str, Dict[str, Any]] = {}
REGISTRY_LOCK = threading.Lock()

# ----------------------------
# Helpers
# ----------------------------
def _load_registry() -> None:
    global JOB_REGISTRY
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                JOB_REGISTRY = json.load(f)
        except Exception:
            JOB_REGISTRY = {}

def _save_registry() -> None:
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(JOB_REGISTRY, f, indent=2)
    os.replace(tmp, REGISTRY_PATH)

def _set_job(job_id: str, **fields: Any) -> None:
    with REGISTRY_LOCK:
        JOB_REGISTRY.setdefault(job_id, {})
        JOB_REGISTRY[job_id].update(fields)
        _save_registry()

def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with REGISTRY_LOCK:
        return JOB_REGISTRY.get(job_id)

def _download_file(url: str, dst_path: str, timeout=(10, 600)) -> None:
    # Stream download to avoid memory blowups
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

# ----------------------------
# Worker
# ----------------------------
def worker_download_only(job_id: str, video_url: str) -> None:
    """
    STEP-1 worker: download only, mark complete.
    In Step-2/3 we'll replace this with ffmpeg + Real-ESRGAN pipeline.
    """
    try:
        _set_job(job_id, status="processing", progress="starting")

        job_dir = os.path.join(JOBS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        input_path = os.path.join(job_dir, "input.mp4")
        output_path = os.path.join(job_dir, "output_4k.mp4")  # placeholder for now

        _set_job(job_id, progress="downloading")
        _download_file(video_url, input_path)

        # For Step-1, we simply copy input -> output to validate download+async+download endpoint.
        _set_job(job_id, progress="finalizing")
        with open(input_path, "rb") as src, open(output_path, "wb") as dst:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                dst.write(buf)

        _set_job(
            job_id,
            status="complete",
            progress="done",
            output_path=output_path,
            download_url=f"/download/{job_id}.mp4",
            finished_at=time.time(),
        )

    except Exception as e:
        _set_job(job_id, status="failed", error=str(e), finished_at=time.time())

# ----------------------------
# API
# ----------------------------
app = FastAPI()

class UpscaleRequest(BaseModel):
    video_url: HttpUrl
    # Keep fields for later, but Step-1 ignores them safely
    scale: Optional[int] = None  # 2 or 4
    target: Optional[str] = "4k"

@app.on_event("startup")
def startup_event():
    _load_registry()

@app.get("/")
def root():
    return {"status": "ok", "service": "runpod-upscale-api", "mode": "async-step-1"}

@app.post("/upscale")
def upscale(req: UpscaleRequest):
    job_id = uuid.uuid4().hex[:10]
    _set_job(
        job_id,
        status="queued",
        progress="queued",
        video_url=str(req.video_url),
        created_at=time.time(),
    )

    t = threading.Thread(
        target=worker_download_only,
        args=(job_id, str(req.video_url)),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return {"job_id": job_id, **job}

@app.get("/download/{job_id}.mp4")
def download(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    if job.get("status") != "complete":
        raise HTTPException(status_code=409, detail=f"job not complete (status={job.get('status')})")

    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="output not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )
