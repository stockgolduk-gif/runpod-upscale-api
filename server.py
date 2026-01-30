from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import os, uuid, subprocess, requests
import torch, cv2
from realesrgan import RealESRGAN

app = FastAPI(root_path="")

BASE_DIR = "/workspace/files"
WORK_DIR = "/workspace/work"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

class UpscaleRequest(BaseModel):
    video_url: HttpUrl
    scale: int = 2  # 2x or 4x


@app.get("/")
def health():
    return {"status": "ok", "service": "runpod-upscale-api"}


@app.post("/upscale-from-url")
def upscale_from_url(payload: UpscaleRequest):
    job_id = uuid.uuid4().hex

    input_video = f"{WORK_DIR}/{job_id}_input.mp4"
    frames_dir = f"{WORK_DIR}/{job_id}_frames"
    up_dir = f"{WORK_DIR}/{job_id}_up"
    output_video = f"{BASE_DIR}/{job_id}_upscaled.mp4"

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(up_dir, exist_ok=True)

    # --- Download video ---
    try:
        r = requests.get(payload.video_url, stream=True, timeout=120)
        r.raise_for_status()
        with open(input_video, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(400, f"Download failed: {e}")

    # --- Extract frames ---
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        f"{frames_dir}/frame_%06d.png"
    ], check=True)

    # --- Load AI upscaler ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealESRGAN(device, scale=payload.scale)
    model.load_weights("RealESRGAN_x4plus.pth", download=True)

    # --- Upscale frames ---
    for f in sorted(os.listdir(frames_dir)):
        img = cv2.imread(f"{frames_dir}/{f}")
        out = model.predict(img)
        cv2.imwrite(f"{up_dir}/{f}", out)

    # --- Rebuild video (Adobe-safe) ---
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", f"{up_dir}/frame_%06d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-crf", "16",
        output_video
    ], check=True)

    return {
        "status": "complete",
        "download_url": f"/download/{os.path.basename(output_video)}"
    }


@app.get("/download/{filename}")
def download(filename: str):
    path = f"{BASE_DIR}/{filename}"
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)
