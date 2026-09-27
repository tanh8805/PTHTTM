"""Phân tích video camera đường phố -> mức cảnh báo ngập.

Dùng model YOLO-seg đã train từ train.ipynb (class: flood, road, car_L0..car_L4).
Chạy thử không cần server:  python analyzer.py duong_dan_video.mp4
"""
import os
import sys
import json
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).parent / "weights" / "best.pt"))
CONF = float(os.getenv("CONF", "0.5"))          # theo F1 curve: 0.5-0.7 là hợp lý
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "1"))  # lấy bao nhiêu khung hình mỗi giây video
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "120"))  # giới hạn số khung hình phân tích mỗi video

# Mức cảnh báo: (mã, tên, tỉ lệ ngập mặt đường tối thiểu, level xe tối thiểu)
# Một khung hình đạt mức nếu thỏa ĐIỀU KIỆN NGẬP HOẶC ĐIỀU KIỆN XE.
LEVELS = [
    (0, "An toàn", 0.00, 0),
    (1, "Chú ý", 0.05, 1),
    (2, "Cảnh báo", 0.30, 2),
    (3, "Nguy hiểm", 0.60, 3),
]
# Mức của cả video = mức cao nhất xuất hiện ở ít nhất MIN_FRAME_FRAC số khung hình
# (tránh vài khung hình nhận nhầm làm báo động giả).
MIN_FRAME_FRAC = float(os.getenv("MIN_FRAME_FRAC", "0.25"))

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Không thấy model: {MODEL_PATH} (đặt best.pt vào backend/weights/ hoặc set MODEL_PATH)")
        _model = YOLO(MODEL_PATH)
    return _model


def class_ids(model):
    """Map class theo tên để không phụ thuộc thứ tự index."""
    names = {i: str(n).lower() for i, n in model.names.items()}
    flood = [i for i, n in names.items() if n == "flood"]
    road = [i for i, n in names.items() if n == "road"]
    car = {i: int(n[-1]) for i, n in names.items() if n.startswith("car_l") and n[-1].isdigit()}
    if not flood:
        raise ValueError(f"Model không có class 'flood': {model.names}")
    return flood, road, car


def analyze_frame(result, flood_ids, road_ids, car_ids):
    h, w = result.orig_shape
    flood_mask = np.zeros((h, w), bool)
    road_mask = np.zeros((h, w), bool)
    cars = []
    if result.masks is not None and len(result.boxes):
        masks = result.masks.data.cpu().numpy() > 0.5
        for m, c, conf in zip(masks, result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if c in flood_ids:
                flood_mask |= m
            elif c in road_ids:
                road_mask |= m
            elif c in car_ids:
                cars.append({"level": car_ids[c], "conf": round(float(conf), 3)})
    road_mask &= ~flood_mask
    flood_px, road_px = int(flood_mask.sum()), int(road_mask.sum())
    # Tỉ lệ ngập so với mặt đường nhìn thấy; thấy quá ít đường (đường ngập hết hoặc model
    # không nhận ra) thì so với cả khung hình, tránh vũng nước nhỏ thành tỉ lệ 100%.
    if road_px >= 0.05 * h * w:
        flood_ratio = flood_px / (flood_px + road_px)
    else:
        flood_ratio = flood_px / (h * w)
    max_car = max((c["level"] for c in cars), default=None)
    level = 0
    for code, _, min_ratio, min_car in LEVELS:
        if flood_ratio >= min_ratio or (max_car is not None and code > 0 and max_car >= min_car):
            level = code
    return {
        "flood_ratio": round(flood_ratio, 3),
        "flood_frame_ratio": round(flood_px / (h * w), 3),
        "cars": len(cars),
        "max_car_level": max_car,
        "level": level,
    }


def read_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Không đọc được video (định dạng không hỗ trợ hoặc file hỏng)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps / SAMPLE_FPS))
    # Video dài: giãn bước để số khung hình không vượt MAX_FRAMES.
    if total and total / step > MAX_FRAMES:
        step = int(np.ceil(total / MAX_FRAMES))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield idx / fps, frame
        idx += 1
    cap.release()


def analyze_video(video_path):
    model = get_model()
    flood_ids, road_ids, car_ids = class_ids(model)
    frames = []
    with _lock:  # model YOLO không an toàn khi nhiều request dùng cùng lúc
        for t, frame in read_frames(video_path):
            r = model.predict(frame, conf=CONF, imgsz=IMG_SIZE, verbose=False)[0]
            frames.append({"time_s": round(t, 1), **analyze_frame(r, flood_ids, road_ids, car_ids)})
            if len(frames) >= MAX_FRAMES:
                break
    if not frames:
        raise ValueError("Video không có khung hình nào")

    n = len(frames)
    counts = [sum(f["level"] >= code for f in frames) for code, *_ in LEVELS]
    level = max(code for code, *_ in LEVELS if counts[code] / n >= MIN_FRAME_FRAC or code == 0)
    ratios = [f["flood_ratio"] for f in frames]
    car_levels = [f["max_car_level"] for f in frames if f["max_car_level"] is not None]
    worst = max(frames, key=lambda f: (f["level"], f["flood_ratio"]))
    return {
        "warning_level": level,
        "warning": LEVELS[level][1],
        "frames_analyzed": n,
        "frames_per_level": {LEVELS[c][1]: sum(f["level"] == c for f in frames) for c in range(len(LEVELS))},
        "flood_ratio_median": round(float(np.median(ratios)), 3),
        "flood_ratio_max": round(float(np.max(ratios)), 3),
        "car_level_max": max(car_levels, default=None),
        "worst_frame_time_s": worst["time_s"],
        "frames": frames,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Dùng: python analyzer.py video.mp4")
    out = analyze_video(sys.argv[1])
    out.pop("frames")
    print(json.dumps(out, ensure_ascii=False, indent=2))
