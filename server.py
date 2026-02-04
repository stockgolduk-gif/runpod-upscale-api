from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import os
import uuid
import threading
import time
import subprocess
import json
import requests

import torch
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# =========================
# App
# =========================
app = FastAPI()

# =========================
# Paths
# =========================
BASE_DIR = "/workspace/files"
WORK_DIR = "/workspace/work"
MODEL_DIR = "/workspace/models"
FFMPEG_PATH = "/usr/bin/ffmpeg"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# Models (stable sources)
# =========================
MODEL_X4_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_X4_PATH = f"{MODEL_DIR}/RealESRGAN_x4plus.pth"

MODEL_X2_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x2plus.pth"
MODEL_X2_PATH = f"{MODEL_DIR}/RealESRGAN_x2plus.pth"

# =========================
# In-memory job store
# =========================
JOBS = {}

# =========================
# Request model
# =========================
class UpscaleRequest(BaseModel):
    video_url: HttpUrl

# =========================
# Utilities
# =========================
def ensure_ffmpeg():
    if not os.path.exists(FFMPEG_PATH):
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)

def ensure_model(url: str, path: str):
    if not os.path.exists(path):
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

def probe_video(path: str):
    cmd = [
        FFMPEG_PATH, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        path
    ]
    out = subprocess.check_output(["ffprobe"] + cmd[1:], text=True)
    data = json.loads(out)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return width, height, fps

# =========================
# Worker
# =========================
def worker(job_id: str, video_url: str):
    try:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["progress"] = "downloading"

        ensure_ffmpeg()

        input_video = f"{WORK_DIR}/{job_id}_input.mp4"
        frames_dir = f"{WORK_DIR}/{job_id}_frames"
        up_dir = f"{WORK_DIR}/{job_id}_up"
        output_video = f"{BASE_DIR}/{job_id}.mp4"

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(up_dir, exist_ok=True)

        # Download
        r = requests.get(video_url, stream=True, timeout=120)
        r.raise_for_status()
        with open(input_video, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

        # Probe video
        JOBS[job_id]["progress"] = "probing video"
        width, height, fps = probe_video(input_video)

        # Decide scale
        if height <= 720:
            scale = 4
            model_path = MODEL_X4_PATH
            model_url = MODEL_X4_URL
        elif height <= 1080:
            scale = 2
            model_path = MODEL_X2_PATH
            model_url = MODEL_X2_URL
        else:
            raise RuntimeError("Input resolution higher than 1080p is not supported")

        ensure_model(model_url, model_path)

        # Extract frames (keep original FPS)
        JOBS[job_id]["progress"] = "extracting frames"
        subprocess.run(
            [
                FFMPEG_PATH, "-y", "-i", input_video,
                f"{frames_dir}/frame_%06d.png"
            ],
            check=True
        )

        # Load model
        JOBS[job_id]["progress"] = "loading model"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )

        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            device=device
        )

        # Upscale frames
        JOBS[job_id]["progress"] = "upscaling frames"
        for fname in sorted(os.listdir(frames_dir)):
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is None:
                continue
            out, _ = upsampler.enhance(img, outscale=scale)
            cv2.imwrite(os.path.join(up_dir, fname), out)

        # Encode to EXACT 4K @ original FPS
        JOBS[job_id]["progress"] = "encoding video"
        subprocess.run(
            [
                FFMPEG_PATH, "-y",
                "-framerate", str(round(fps)),
                "-i", f"{up_dir}/frame_%06d.png",
                "-vf", "scale=3840:2160",
                "-c:v", "libx264",
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",
                "-crf", "16",
                output_video
            ],
            check=True
        )

        JOBS[job_id]["status"] = "complete"
        JOBS[job_id]["progress"] = "done"
        JOBS[job_id]["download_url"] = f"/download/{job_id}.mp4"
        JOBS[job_id]["finished_at"] = time.time()

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["finished_at"] = time.time()

# =========================
# API
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "runpod-upscale-api",
        "mode": "dynamic-x2-x4-4k-60fps"
    }

@app.post("/upscale")
def upscale(req: UpscaleRequest):
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": "queued",
        "video_url": str(req.video_url),
        "created_at": time.time()
    }

    t = threading.Thread(
        target=worker,
        args=(job_id, str(req.video_url)),
        daemon=True
    )
    t.start()

    return {
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JOBS[job_id]

@app.get("/download/{filename}")
def download(filename: str):
    path = f"{BASE_DIR}/{filename}"
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)
