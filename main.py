from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


MODEL_PATH = Path("best.pt")
VIDEO_PATH = Path("flood_2.mp4")

DEVICE = 0 if torch.cuda.is_available() else "cpu"


model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))


while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.25,
        device=DEVICE,
        retina_masks=True,
        verbose=False
    )

    result = results[0]

    # Có segmentation mask
    if result.masks is not None:

        for i, cls in enumerate(result.boxes.cls):

            # Chỉ lấy FLOOD
            if int(cls) != 0:
                continue

            mask = result.masks.data[i].cpu().numpy()

            # Mask về kích thước frame
            mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0])
            )

            mask = mask > 0.5

            # Tô đỏ vùng FLOOD
            overlay = frame.copy()
            overlay[mask] = (0, 0, 255)

            frame = cv2.addWeighted(
                overlay,
                0.4,
                frame,
                0.6,
                0
            )

            # Viền FLOOD
            contours, _ = cv2.findContours(
                mask.astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                frame,
                contours,
                -1,
                (0, 0, 255),
                3
            )

            # Confidence
            conf = float(result.boxes.conf[i])

            cv2.putText(
                frame,
                f"FLOOD {conf:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("Flood Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()