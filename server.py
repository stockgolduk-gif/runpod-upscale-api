from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import os
import uuid
import json
from typing import Dict

app = FastAPI()

# Directory where uploaded files are stored
BASE_DIR = "/workspace/files"
os.makedirs(BASE_DIR, exist_ok=True)

# Directory where job status files are stored
JOBS_DIR = "/workspace/jobs"
os.makedirs(JOBS_DIR, exist_ok=True)


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


# -----------------------------
# STEP 1: Add job system (no GPU yet)
# -----------------------------

def set_job_status(job_id: str, data: Dict):
    path = os.path.join(JOBS_DIR, f"{job_id}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_job_status(job_id: str):
    path = os.path.join(JOBS_DIR, f"{job_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def dummy_process(job_id: str, filename: str):
    """
    Placeholder for future processing.
    For now, it just waits briefly and marks complete.
    """
    set_job_status(job_id, {
        "state": "processing",
        "progress": 10,
        "input_file": filename
    })

    import time
    time.sleep(10)

    set_job_status(job_id, {
        "state": "complete",
        "progress": 100,
        "input_file": filename,
        "output_file": filename
    })


@app.post("/upscale")
def upscale(filename: str, background_tasks: BackgroundTasks):
    """
    Start an upscale job using an already-uploaded file.
    (Step 1: no real upscaling yet)
    """
    input_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(input_path):
        return {"error": "File not found"}

    job_id = uuid.uuid4().hex

    set_job_status(job_id, {
        "state": "queued",
        "progress": 0,
        "input_file": filename
    })

    background_tasks.add_task(dummy_process, job_id, filename)

    return {
        "job_id": job_id,
        "status_url": f"/status/{job_id}"
    }


@app.get("/status/{job_id}")
def status(job_id: str):
    """
    Check job status.
    """
    data = get_job_status(job_id)
    if not data:
        return {"error": "Job not found"}
    return data
