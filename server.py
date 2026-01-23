from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import uuid

app = FastAPI()

BASE_DIR = "/workspace/files"
os.makedirs(BASE_DIR, exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(BASE_DIR, name)

    with open(path, "wb") as f:
        f.write(await file.read())

    return {
        "filename": name,
        "download_url": f"/download/{name}"
    }

@app.get("/download/{filename}")
def download(filename: str):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(path, filename=filename)
