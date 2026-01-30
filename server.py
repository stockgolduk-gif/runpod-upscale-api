from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import os
import uuid
import urllib.request
import urllib.parse

# Keep this (fixes RunPod proxy path / docs issues)
app = FastAPI(root_path="")

# Directory where files are stored
BASE_DIR = "/workspace/files"
os.makedirs(BASE_DIR, exist_ok=True)


class UpscaleFromUrlRequest(BaseModel):
    video_url: HttpUrl


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "source": "github",
        "service": "runpod-upscale-api"
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(BASE_DIR, filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    return {
        "filename": filename,
        "download_url": f"/download/{filename}"
    }


@app.post("/upscale-from-url")
def upscale_from_url(payload: UpscaleFromUrlRequest):
    """
    Step 1: Accept a video URL, download it to /workspace/files,
    and return a download link. (No upscaling yet.)
    """
    url = str(payload.video_url)

    # Try to infer an extension from the URL path; fallback to .mp4
    parsed = urllib.parse.urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    if not ext or len(ext) > 6:
        ext = ".mp4"

    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(BASE_DIR, filename)

    try:
        # Download to disk
        with urllib.request.urlopen(url, timeout=120) as r:
            if getattr(r, "status", 200) >= 400:
                raise HTTPException(status_code=400, detail=f"Failed to download. HTTP {r.status}")

            with open(path, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

    except HTTPException:
        raise
    except Exception as e:
        # Clean up partial file if any
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    return {
        "status": "downloaded",
        "filename": filename,
        "download_url": f"/download/{filename}"
    }


@app.get("/download/{filename}")
def download(filename: str):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        return {"error": "File not found"}

    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=filename
    )
