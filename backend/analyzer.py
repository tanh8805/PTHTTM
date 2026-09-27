"""Phân tích video camera đường phố -> video segmentation + mức cảnh báo ngập theo từng giai đoạn.

Dùng model YOLO-seg đã train từ train.ipynb (class: flood, road, car_L0..car_L4).
Chạy thử không cần server:  python analyzer.py video.mp4 [video_ket_qua.mp4]
"""
import os
import sys
import json
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).parent / "weights" / "best.pt"))
CONF = float(os.getenv("CONF", "0.5"))                # theo F1 curve: 0.5-0.7 là hợp lý
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
INFER_FPS = float(os.getenv("INFER_FPS", "5"))        # số lần chạy model mỗi giây video; khung hình ở giữa dùng lại mask gần nhất
SEGMENT_SECONDS = float(os.getenv("SEGMENT_SECONDS", "2"))  # độ dài mỗi đoạn để tính mức cảnh báo
MIN_FRAME_FRAC = float(os.getenv("MIN_FRAME_FRAC", "0.5"))  # mức của đoạn = mức cao nhất mà >= 50% khung hình trong đoạn đạt
OUT_WIDTH = int(os.getenv("OUT_WIDTH", "960"))        # chiều rộng video kết quả (thu nhỏ cho nhẹ)
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "600"))  # chỉ xử lý tối đa 10 phút đầu

# Mức cảnh báo: (mã, tên, tỉ lệ ngập mặt đường tối thiểu, level xe tối thiểu, màu BGR)
# Một khung hình đạt mức nếu thỏa ĐIỀU KIỆN NGẬP HOẶC ĐIỀU KIỆN XE.
LEVELS = [
    (0, "An toàn", 0.00, 0, (70, 160, 40)),
    (1, "Chú ý", 0.05, 1, (0, 185, 235)),
    (2, "Cảnh báo", 0.30, 2, (0, 120, 245)),
    (3, "Nguy hiểm", 0.60, 3, (40, 30, 210)),
]
# Màu từng class (BGR) khi vẽ mask
CLASS_COLORS = {
    "flood": (230, 120, 20),
    "road": (150, 150, 150),
    "car_L0": (80, 200, 80),
    "car_L1": (60, 220, 200),
    "car_L2": (0, 215, 255),
    "car_L3": (0, 140, 255),
    "car_L4": (40, 40, 230),
}

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Không thấy model: {MODEL_PATH} (đặt best.pt vào backend/weights/ hoặc set MODEL_PATH)")
        _model = YOLO(MODEL_PATH)
    return _model


def class_map(model):
    """Map class theo tên để không phụ thuộc thứ tự index: id -> (loại, level xe)."""
    cmap = {}
    for i, name in model.names.items():
        n = str(name).lower()
        if n in ("flood", "road"):
            cmap[int(i)] = (n, None)
        elif n.startswith("car_l") and n[-1].isdigit():
            cmap[int(i)] = ("car", int(n[-1]))
    if not any(kind == "flood" for kind, _ in cmap.values()):
        raise ValueError(f"Model không có class 'flood': {model.names}")
    return cmap


# ---------------- Tính mức cảnh báo ----------------

def parse_result(result, cmap):
    """Kết quả YOLO -> danh sách instance {kind, level, conf, mask, box} (mask cùng kích thước khung hình)."""
    h, w = result.orig_shape
    instances = []
    if result.masks is None or not len(result.boxes):
        return instances
    masks = result.masks.data.cpu().numpy() > 0.5
    for m, c, conf, box in zip(masks, result.boxes.cls.cpu().numpy().astype(int),
                               result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
        if c not in cmap:
            continue
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        kind, level = cmap[c]
        instances.append({"kind": kind, "level": level, "conf": float(conf), "mask": m, "box": box.astype(int)})
    return instances


def level_of(flood_ratio, max_car):
    level = 0
    for code, _, min_ratio, min_car, _ in LEVELS[1:]:
        if flood_ratio >= min_ratio or (max_car is not None and max_car >= min_car):
            level = code
    return level


def frame_stats(instances, h, w):
    flood = np.zeros((h, w), bool)
    road = np.zeros((h, w), bool)
    cars = []
    for inst in instances:
        if inst["kind"] == "flood":
            flood |= inst["mask"]
        elif inst["kind"] == "road":
            road |= inst["mask"]
        else:
            cars.append(inst["level"])
    road &= ~flood
    flood_px, road_px = int(flood.sum()), int(road.sum())
    # Tỉ lệ ngập so với mặt đường nhìn thấy; thấy quá ít đường (đường ngập hết hoặc model
    # không nhận ra) thì so với cả khung hình, tránh vũng nước nhỏ thành tỉ lệ 100%.
    road_visible = road_px >= 0.05 * h * w
    flood_ratio = flood_px / (flood_px + road_px) if road_visible else flood_px / (h * w)
    max_car = max(cars, default=None)
    return {
        "flood_ratio": round(flood_ratio, 3),
        "road_visible": road_visible,
        "cars": len(cars),
        "max_car_level": max_car,
        "level": level_of(flood_ratio, max_car),
    }


def segment_level(stats):
    n = len(stats)
    return max(code for code, *_ in LEVELS if code == 0 or sum(s["level"] >= code for s in stats) / n >= MIN_FRAME_FRAC)


def max_or_none(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


# ---------------- Vẽ lên video ----------------

def _find_font():
    env = os.getenv("FONT_PATH", "")
    if env and Path(env).exists():
        return env
    # matplotlib (luôn được cài cùng ultralytics) có sẵn DejaVu Sans, hiển thị đủ dấu tiếng Việt
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        if p.exists():
            return str(p)
    except Exception:
        pass
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    return next((c for c in candidates if Path(c).exists()), None)


def _font(size):
    if FONT_FILE:
        return ImageFont.truetype(FONT_FILE, size)
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


FONT_FILE = _find_font()


def vn_text(s):
    """Không có font tiếng Việt thì bỏ dấu để chữ không bị lỗi ô vuông."""
    if FONT_FILE:
        return s
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


@lru_cache(maxsize=512)
def banner(width, level, line2):
    """Thanh cảnh báo phía trên video (PIL để hiển thị được tiếng Việt)."""
    scale = width / 960
    height = int(70 * scale)
    b, g, r = LEVELS[level][4]
    img = Image.new("RGB", (width, height), (r, g, b))
    draw = ImageDraw.Draw(img)
    big = _font(int(30 * scale))
    small = _font(int(18 * scale))
    draw.text((int(14 * scale), int(5 * scale)), vn_text(f"MỨC {level}: {LEVELS[level][1].upper()}"), font=big, fill=(255, 255, 255))
    draw.text((int(14 * scale), int(42 * scale)), vn_text(line2), font=small, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def build_overlay(instances, h, w):
    """Chuẩn bị lớp màu mask + viền + nhãn xe; dùng lại cho các khung hình tới lần chạy model kế tiếp."""
    color = np.zeros((h, w, 3), np.uint8)
    area = np.zeros((h, w), bool)
    contours, labels = [], []
    order = {"road": 0, "flood": 1, "car": 2}  # vẽ xe sau cùng để nằm trên nước/đường
    for inst in sorted(instances, key=lambda i: order[i["kind"]]):
        name = f"car_L{inst['level']}" if inst["kind"] == "car" else inst["kind"]
        col = CLASS_COLORS[name]
        color[inst["mask"]] = col
        area |= inst["mask"]
        cs, _ = cv2.findContours(inst["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.append((cs, col))
        if inst["kind"] == "car":
            labels.append((f"L{inst['level']} {inst['conf']:.2f}", tuple(inst["box"][:2]), col))
    return color, area, contours, labels


def draw_legend(img):
    h = img.shape[0]
    y0 = h - 10 - 20 * len(CLASS_COLORS)
    cv2.rectangle(img, (6, y0 - 6), (118, h - 6), (0, 0, 0), -1)
    for k, (name, col) in enumerate(CLASS_COLORS.items()):
        y = y0 + 20 * k
        cv2.rectangle(img, (12, y + 2), (26, y + 14), col, -1)
        cv2.putText(img, name, (32, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(frame, overlay, level, stats):
    out = frame.copy()
    if overlay is not None:
        color, area, contours, labels = overlay
        out[area] = (out[area] * 0.5 + color[area] * 0.5).astype(np.uint8)
        for cs, col in contours:
            cv2.drawContours(out, cs, -1, col, 2)
        for text, (x, y), col in labels:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y = max(y, th + 4)
            cv2.rectangle(out, (x, y - th - 4), (x + tw + 4, y), col, -1)
            cv2.putText(out, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    draw_legend(out)
    car = "không có xe" if stats["max_car_level"] is None else f"L{stats['max_car_level']}"
    base = "mặt đường" if stats["road_visible"] else "khung hình (không thấy đường)"
    top = banner(out.shape[1], level, f"Ngập {stats['flood_ratio']:.0%} {base} · Xe ngập cao nhất: {car}")
    return np.vstack([top, out])


def open_writer(path, w, h, fps):
    """Ghi H.264 để trình duyệt phát được (mp4v của OpenCV không phát trên web)."""
    writer = imageio_ffmpeg.write_frames(
        str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="bgr24", quality=None, macro_block_size=2,
        output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"])
    writer.send(None)
    return writer


# ---------------- Xử lý cả video ----------------

def process_video(video_path, out_path, progress=None):
    """Tạo video segmentation có thanh cảnh báo; trả về mức cảnh báo tổng + từng giai đoạn."""
    model = get_model()
    cmap = class_map(model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Không đọc được video (định dạng không hỗ trợ hoặc file hỏng)")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:
        fps = 25.0
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = min(OUT_WIDTH, src_w) // 2 * 2
    h = round(src_h * w / src_w) // 2 * 2
    max_frames = int(MAX_SECONDS * fps)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), max_frames) or max_frames
    infer_every = max(1, round(fps / INFER_FPS))
    seg_len = max(infer_every, round(SEGMENT_SECONDS * fps))

    frames, segments = [], []
    buf, seg_stats = [], []   # khung hình của đoạn hiện tại (chờ biết mức của đoạn rồi mới ghi)
    overlay, stats = None, None
    writer = open_writer(out_path, w, h + banner(w, 0, "").shape[0], fps)
    idx = seg_start = 0

    def flush():
        nonlocal seg_start
        if not buf:
            return
        cur = seg_stats or [stats]
        level = segment_level(cur)
        segments.append({
            "start_s": round(seg_start / fps, 2), "end_s": round(idx / fps, 2),
            "level": level, "warning": LEVELS[level][1],
            "flood_ratio_max": max(s["flood_ratio"] for s in cur),
            "car_level_max": max_or_none(s["max_car_level"] for s in cur),
        })
        for frame, ov, st in buf:
            writer.send(np.ascontiguousarray(render_frame(frame, ov, level, st)))
        buf.clear()
        seg_stats.clear()
        seg_start = idx

    with _lock:  # model YOLO không an toàn khi nhiều video dùng cùng lúc
        try:
            while idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                if idx % infer_every == 0:
                    r = model.predict(frame, conf=CONF, imgsz=IMG_SIZE, retina_masks=True, verbose=False)[0]
                    instances = parse_result(r, cmap)
                    stats = {"time_s": round(idx / fps, 2), **frame_stats(instances, h, w)}
                    frames.append(stats)
                    seg_stats.append(stats)
                    overlay = build_overlay(instances, h, w)
                buf.append((frame, overlay, stats))
                idx += 1
                if idx % seg_len == 0:
                    flush()
                if progress:
                    progress(min(idx / total, 0.99))
            flush()
        finally:
            cap.release()
            writer.close()
    if not frames:
        raise ValueError("Video không có khung hình nào")

    # Gộp các đoạn liền nhau cùng mức thành một giai đoạn
    phases = []
    for s in segments:
        if phases and phases[-1]["level"] == s["level"]:
            p = phases[-1]
            p["end_s"] = s["end_s"]
            p["flood_ratio_max"] = max(p["flood_ratio_max"], s["flood_ratio_max"])
            p["car_level_max"] = max_or_none([p["car_level_max"], s["car_level_max"]])
        else:
            phases.append(dict(s))
    level = max(s["level"] for s in segments)
    return {
        "warning_level": level,
        "warning": LEVELS[level][1],
        "duration_s": round(idx / fps, 2),
        "frames_analyzed": len(frames),
        "seconds_per_level": {LEVELS[c][1]: round(sum(p["end_s"] - p["start_s"] for p in phases if p["level"] == c), 2)
                              for c in range(len(LEVELS))},
        "flood_ratio_max": max(f["flood_ratio"] for f in frames),
        "car_level_max": max_or_none(f["max_car_level"] for f in frames),
        "phases": phases,
        "segments": segments,
        "frames": frames,
    }


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Dùng: python analyzer.py video.mp4 [video_ket_qua.mp4]")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) == 3 else src.with_name(src.stem + "_result.mp4")
    out = process_video(src, dst)
    out.pop("frames")
    out.pop("segments")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("Video kết quả:", dst)
