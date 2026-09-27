# Backend cảnh báo ngập từ video

Upload video camera đường phố, nhận lại mức cảnh báo 0–3. Dùng `best.pt` từ `train.ipynb`.

## Chạy

```bash
cd backend
pip install -r requirements.txt
# chép best.pt (output notebook train trên Kaggle) vào backend/weights/best.pt
uvicorn app:app --host 0.0.0.0 --port 8000
```

- Trình duyệt: http://localhost:8000 → chọn video → **Phân tích**
- API docs: http://localhost:8000/docs
- Dòng lệnh:
  ```bash
  curl -F "file=@video.mp4" http://localhost:8000/analyze
  curl -F "file=@video.mp4" "http://localhost:8000/analyze?include_frames=true"   # kèm kết quả từng khung hình
  ```
- Không cần server: `python analyzer.py video.mp4`

## Cách tính mức cảnh báo

1. Lấy 1 khung hình/giây (tối đa 120 khung hình), chạy YOLO-seg với conf 0.5.
2. Mỗi khung hình:
   - `flood_ratio` = diện tích flood / (flood + road). Thấy ít đường (< 5% khung hình) thì chia cho cả khung hình.
   - `max_car_level` = level cao nhất của xe (car_L0…car_L4, càng cao nước càng sâu).
3. Mức của khung hình: đạt **một trong hai** điều kiện.

   | Mức | Tên | flood_ratio ≥ | hoặc xe ≥ |
   |---|---|---|---|
   | 0 | An toàn | – | – |
   | 1 | Chú ý | 0.05 | L1 |
   | 2 | Cảnh báo | 0.30 | L2 |
   | 3 | Nguy hiểm | 0.60 | L3 |

4. Mức của cả video = mức cao nhất xuất hiện ở ít nhất 25% số khung hình, để tránh báo động giả do vài khung hình nhận nhầm.

Ngưỡng nằm trong `LEVELS` của `analyzer.py`, sửa theo thực tế. Các tham số chỉnh qua biến môi trường: `MODEL_PATH`, `CONF`, `SAMPLE_FPS`, `MAX_FRAMES`, `MIN_FRAME_FRAC`, `MAX_UPLOAD_MB`.

## Ví dụ kết quả

```json
{
  "filename": "cam1.mp4",
  "warning_level": 2,
  "warning": "Cảnh báo",
  "frames_analyzed": 30,
  "frames_per_level": {"An toàn": 2, "Chú ý": 6, "Cảnh báo": 20, "Nguy hiểm": 2},
  "flood_ratio_median": 0.42,
  "flood_ratio_max": 0.66,
  "car_level_max": 3,
  "worst_frame_time_s": 17.0
}
```
