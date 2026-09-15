"""
FastAPI backend: nhận video upload, chạy model segmentation (best.pt)
để nhận diện vùng ngập úng (class FLOOD), tô đỏ + viền rồi xuất video kết quả
kèm thống kê và mức độ cảnh báo.

Chạy:  .venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
"""

import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- config
CONF = 0.25            # confidence threshold (giống main.py)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ---------------------------------------------------------------- model
app = FastAPI(title="Flood Detection API - Cảnh báo ngập úng")
_model = None
_jobs: dict[str, dict] = {}          # job_id -> state
_jobs_lock = threading.Lock()


def get_model() -> YOLO:
    global _model
    if _model is None:
        print(f"[INFO] Loading model {MODEL_PATH} on {DEVICE} ...")
        _model = YOLO(str(MODEL_PATH))
    return _model


def _make_writer(path: Path, fps: float, w: int, h: int):
    """Tạo VideoWriter; ưu tiên H.264 (avc1) để trình duyệt xem được, fallback mp4v."""
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if writer.isOpened():
            return writer
    return None

# ---------------------------------------------------------------- processing
def process_video(job_id: str, src_path: Path, dst_path: Path) -> None:
    state = {
        "status": "processing",
        "progress": 0,
        "message": "Đang xử lý video...",
    }
    with _jobs_lock:
        _jobs[job_id].update(state)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        with _jobs_lock:
            _jobs[job_id].update(
                {"status": "error", "message": "Không đọc được video", "progress": 0}
            )
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = _make_writer(dst_path, fps, width, height)

    model = get_model()

    frame_idx = 0
    flood_frames = 0
    flood_area_sum = 0.0            # tổng tỉ lệ diện tích ngập trên các frame có ngập
    episodes = 0
    was_flooding = False
    episode_starts = []
    last_update = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_flooding = False

        results = model.predict(
            frame,
            conf=CONF,
            device=DEVICE,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]

        if result.masks is not None and result.boxes is not None:
            for i, cls in enumerate(result.boxes.cls):
                # Chỉ lấy class FLOOD (0)
                if int(cls) != 0:
                    continue

                mask = result.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = mask > 0.5

                frame_flooding = True
                flood_area_sum += float(mask.sum()) / float(frame.shape[0] * frame.shape[1])

                overlay = frame.copy()
                overlay[mask] = (0, 0, 255)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                contours, _ = cv2.findContours(
                    mask.astype("uint8"),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

                conf = float(result.boxes.conf[i])
                cv2.putText(
                    frame,
                    f"FLOOD {conf:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

        if frame_flooding:
            flood_frames += 1
            if not was_flooding:
                episodes += 1
                episode_starts.append(round(frame_idx / fps, 2))
            was_flooding = True
        else:
            was_flooding = False

        if writer is not None:
            writer.write(frame)

        frame_idx += 1

        # cập nhật tiến độ theo thời gian thực
        now = time.time()
        if now - last_update > 0.5:
            last_update = now
            progress = (
                int(frame_idx / total_frames * 100) if total_frames else 50
            )
            with _jobs_lock:
                _jobs[job_id].update(
                    {"progress": min(progress, 99), "frames_done": frame_idx}
                )

    cap.release()
    if writer is not None:
        writer.release()
    elif dst_path.exists():
        dst_path.unlink()

    # ---------------- thống kê & cảnh báo
    flood_ratio = (flood_frames / frame_idx * 100) if frame_idx else 0.0
    avg_flood_area = (flood_area_sum / flood_frames) if flood_frames else 0.0

    if flood_ratio >= 50 or avg_flood_area >= 0.30:
        level, detail = 3, "NGUY HIỂM — ngập diện rộng, cần sơ tán"
    elif flood_ratio >= 20 or avg_flood_area >= 0.10:
        level, detail = 2, "CẢNH BÁO — có nguy cơ ngập úng"
    elif flood_frames > 0:
        level, detail = 1, "LƯU Ý — phát hiện điểm ngập nhỏ"
    else:
        level, detail = 0, "AN TOÀN — không phát hiện ngập"

    with _jobs_lock:
        _jobs[job_id].update(
            {
                "status": "done",
                "progress": 100,
                "message": "Hoàn tất",
                "result": {
                    "output_url": f"/output/{dst_path.name}",
                    "output_name": dst_path.name,
                    "total_frames": frame_idx,
                    "fps": round(fps, 2),
                    "resolution": f"{width}x{height}",
                    "flood_frames": flood_frames,
                    "flood_ratio_percent": round(flood_ratio, 2),
                    "avg_flood_area_ratio": round(avg_flood_area, 4),
                    "episodes": episodes,
                    "episode_starts": episode_starts[:20],
                    "warning_level": level,
                    "warning_text": detail,
                },
            }
        )


# ---------------------------------------------------------------- API routes
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ: {ext or 'không rõ'}. "
                   f"Cho phép: {', '.join(sorted(ALLOWED_EXT))}",
        )

    job_id = uuid.uuid4().hex[:12]
    src_path = UPLOAD_DIR / f"{job_id}{ext}"
    dst_path = OUTPUT_DIR / f"{job_id}_out.mp4"

    with src_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Đang chờ xử lý...",
            "filename": file.filename,
            "result": None,
        }

    thread = threading.Thread(
        target=process_video, args=(job_id, src_path, dst_path), daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    return job


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# mount tĩnh: frontend + video kết quả
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)