from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import uuid

# IMPORTANT: root_path is required for RunPod proxy
app = FastAPI(root_path="")

# Directory where uploaded files are stored
BASE_DIR = "/workspace/files"
os.makedirs(BASE_DIR, exist_ok=True)


@app.get("/")
def health_check():
    """
    Simple sanity check to confirm the server is running
    and that this code is coming from GitHub.
    """
    return {
        "status": "ok",
        "source": "github",
        "service": "runpod-upscale-api"
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a file (image or video).
    The file is saved with a UUID filename.
    """
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(BASE_DIR, filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    return {
        "filename": filename,
        "download_url": f"/download/{filename}"
    }


@app.get("/download/{filename}")
def download(filename: str):
    """
    Download a previously uploaded file.
    """
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        return {"error": "File not found"}

    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=filename
    )
