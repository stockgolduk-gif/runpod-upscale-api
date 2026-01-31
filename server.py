import os
import uuid
import json
import time
import threading
import subprocess
from typing import Dict, Any, Optional

import requests
import cv2
import torch
from realesrgan import RealESRGAN

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

# ============================================================
# App
# ============================================================

app = FastAPI(root_path="")

# ============================================================
# Paths
# ============================================================

BASE_DIR = "/workspace"
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
REGISTRY_PATH = os.path.join(JOBS_DIR, "registry.json")

os.makedirs(JOBS_DIR, exist_ok=True)

# ============================================================
# Job Registry (in-memory + disk mirror)
# ============================================================

JOB_REGISTRY: Dict[str, Dict[str, Any]] = {}
REGISTRY_LOCK = threading.Lock()


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


# ============================================================
# Helpers
# ============================================================

def _download_file(url: str, dst_path: str, timeout=(10, 600)) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


# ============================================================
# Worker — REAL pipeline (Step 2)
# ============================================================

def worker_upscale_basic(job_id: str, video_url: str) -> None:
    """
    STEP 2:
    - Download video
    - Extract frames
    - Real-ESRGAN upscale (fixed scale=2 for now)
    - Rebuild video
    """

    try:
        _set_job(job_id, status="processing", progress="starting")

        job_dir = os.path.join(JOBS_DIR, job_id)
        frames_dir = os.path.join(job_dir, "frames")
        up_dir = os.path.join(job_dir, "up")

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(up_dir, exist_ok=True)

        input_video = os.path.join(job_dir, "input.mp4")
        output_video = os.path.join(job_dir, "output_4k.mp4")

        # -----------------------
        # Download
        # -----------------------
        _set_job(job_id, progress="downloading")
        _download_file(video_url, input_video)

        # -----------------------
        # Extract frames
        # -----------------------
        _set_job(job_id, progress="extracting frames")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_video,
                f"{frames_dir}/frame_%06d.png",
            ],
            check=True,
        )

        # -----------------------
        # AI Upscale (fixed x2 for Step 2)
        # -----------------------
        _set_job(job_id, progress="upscaling frames")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RealESRGAN(device, scale=2)
        model.load_weights("RealESRGAN_x4plus.pth", download=True)

        frame_files = sorted(os.listdir(frames_dir))
        total = len(frame_files)

        for idx, name in enumerate(frame_files, start=1):
            img_path = os.path.join(frames_dir, name)
            out_path = os.path.join(up_dir, name)

            img = cv2.imread(img_path)
            out = model.predict(img)
            cv2.imwrite(out_path, out)

            if idx % 25 == 0:
                _set_job(job_id, progress=f"upscaling {idx}/{total}")

        # -----------------------
        # Rebuild video
        # -----------------------
        _set_job(job_id, progress="encoding video")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "30",
                "-i",
                f"{up_dir}/frame_%06d.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "16",
                output_video,
            ],
            check=True,
        )

        # -----------------------
        # Complete
        # -----------------------
        _set_job(
            job_id,
            status="complete",
            progress="done",
            output_path=output_video,
            download_url=f"/download/{job_id}.mp4",
            finished_at=time.time(),
        )

    except Exception as e:
        _set_job(
            job_id,
            status="failed",
            error=str(e),
            finished_at=time.time(),
        )


# ============================================================
# API
# ============================================================

class UpscaleRequest(BaseModel):
    video_url: HttpUrl


@app.on_event("startup")
def startup_event():
    _load_registry()


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "runpod-upscale-api",
        "mode": "async-step-2",
    }


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
        target=worker_upscale_basic,
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
        raise HTTPException(
            status_code=409,
            detail=f"job not complete (status={job.get('status')})",
        )

    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="output not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )
