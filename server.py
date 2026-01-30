from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import os
import uuid
import requests
from urllib.parse import urlparse

app = FastAPI(root_path="")

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


@app.post("/upscale-from-url")
def upscale_from_url(payload: UpscaleFromUrlRequest):
    url = str(payload.video_url)

    # Infer extension (fallback to .mp4)
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1]
    if not ext:
        ext = ".mp4"

    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(BASE_DIR, filename)

    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()

        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "downloaded",
        "filename": filename,
        "download_url": f"/download/{filename}"
    }


@app.get("/download/{filename}")
def download(filename: str):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=filename
    )
