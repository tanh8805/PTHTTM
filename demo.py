"""
Demo xử lý video KHÔNG cần FastAPI/uvicorn.
Nhận diện vùng ngập (class FLOOD) bằng best.pt, tô đỏ + viền, lưu video kết quả.

Chạy:  python demo.py   (sửa VIDEO_PATH bên dưới nếu muốn đổi video)
Cần:   pip install ultralytics opencv-python
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# ====== ĐIỀN ĐƯỜNG DẪN TRỰC TIẾP Ở ĐÂY ======
VIDEO_PATH = BASE_DIR / "uploads" / "3070e7a63ba0.mp4"
OUTPUT_PATH = BASE_DIR / "outputs" / "demo_result.mp4"
MODEL_PATH = BASE_DIR / "best.pt"
CONF = 0.25
SHOW = True  # True: hiện cửa sổ xem trực tiếp (nhấn q để dừng)
# ============================================


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise SystemExit(f"Không đọc được video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = flood_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, conf=CONF, retina_masks=True, verbose=False)[0]
        flooding = False
        if result.masks is not None and result.boxes is not None:
            for i, cls in enumerate(result.boxes.cls):
                if int(cls) != 0:  # chỉ lấy class FLOOD
                    continue
                mask = result.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (w, h)) > 0.5
                flooding = True

                overlay = frame.copy()
                overlay[mask] = (0, 0, 255)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
                cv2.putText(frame, f"FLOOD {float(result.boxes.conf[i]):.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        flood_frames += flooding
        writer.write(frame)
        idx += 1
        print(f"\rFrame {idx}/{total}", end="", flush=True)

        if SHOW:
            cv2.imshow("Flood demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nXong! Frame có ngập: {flood_frames}/{idx}")
    print(f"Video kết quả: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
