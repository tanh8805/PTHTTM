"""Backend đơn giản: upload video -> video segmentation + mức cảnh báo ngập theo từng giai đoạn.

Chạy:  uvicorn app:app --host 0.0.0.0 --port 8000
Mở http://localhost:8000 để upload bằng trình duyệt, hoặc xem API ở http://localhost:8000/docs
"""
import os
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

import analyzer

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).parent / "outputs"))
STATIC_DIR = Path(__file__).parent / "static"

JOBS = {}                                    # job_id -> trạng thái (mất khi tắt server)
executor = ThreadPoolExecutor(max_workers=1)  # xử lý lần lượt từng video


@asynccontextmanager
async def lifespan(app):
    analyzer.get_model()  # nạp model 1 lần khi khởi động, báo lỗi sớm nếu thiếu best.pt
    yield


app = FastAPI(title="Cảnh báo ngập từ video", lifespan=lifespan)


def run_job(job_id, video_path):
    job = JOBS[job_id]
    job["status"] = "running"
    out_dir = OUTPUT_DIR / job_id
    try:
        result = analyzer.process_video(video_path, out_dir / "result.mp4",
                                        progress=lambda p: job.update(progress=round(p, 3)))
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        result.pop("frames")
        job.update(status="done", progress=1.0, result=result, video_url=f"/jobs/{job_id}/video")
    except ValueError as e:
        job.update(status="error", error=str(e))
    except Exception as e:
        job.update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        os.remove(video_path)


@app.get("/health")
def health():
    return {"status": "ok", "model": analyzer.MODEL_PATH}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Chỉ nhận video {sorted(VIDEO_EXTS)}")
    job_id = uuid.uuid4().hex[:12]
    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True)
    video_path = out_dir / f"input{ext}"
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with open(video_path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                os.remove(video_path)
                raise HTTPException(413, f"Video quá {MAX_UPLOAD_MB}MB")
            f.write(chunk)
    JOBS[job_id] = {"job_id": job_id, "filename": file.filename, "status": "queued", "progress": 0.0}
    executor.submit(run_job, job_id, video_path)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Không có job này")
    return JOBS[job_id]


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str):
    if JOBS.get(job_id, {}).get("status") != "done":
        raise HTTPException(404, "Video chưa xử lý xong")
    return FileResponse(OUTPUT_DIR / job_id / "result.mp4", media_type="video/mp4")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
