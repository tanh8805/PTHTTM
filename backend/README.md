# Backend cảnh báo ngập từ video

Upload video camera đường phố, nhận lại:
- **Video segmentation**: mask từng class (flood, road, car_L0…car_L4) vẽ lên video, kèm thanh cảnh báo phía trên đổi màu theo mức.
- **Mức cảnh báo theo từng giai đoạn** (ví dụ 0–12s An toàn, 12–30s Cảnh báo…), cùng mức cao nhất của cả video.

Model là `best.pt` từ `train.ipynb`.

## Chạy

```bash
cd backend
pip install -r requirements.txt
# chép best.pt (output notebook train trên Kaggle) vào backend/weights/best.pt
uvicorn app:app --host 0.0.0.0 --port 8000
```

- **Trình duyệt**: mở http://localhost:8000, chọn video, bấm **Phân tích** rồi chờ thanh tiến trình chạy xong. Trang hiện:
  - video segmentation;
  - thanh thời gian tô màu theo mức, bấm vào để tua tới;
  - bảng các giai đoạn;
  - ô "Đang xem" cho biết mức của đoạn đang phát.
- **Dòng lệnh, không cần server**: `python analyzer.py video.mp4`. Lệnh in kết quả JSON và tạo `video_result.mp4`.
- **API** (xem thêm http://localhost:8000/docs):
  ```bash
  curl -F "file=@video.mp4" http://localhost:8000/analyze    # -> {"job_id": "..."}
  curl http://localhost:8000/jobs/<job_id>                   # status: queued/running/done/error, progress, result
  curl -o ket_qua.mp4 http://localhost:8000/jobs/<job_id>/video
  ```
  Kết quả đầy đủ, gồm số liệu từng khung hình, nằm ở `backend/outputs/<job_id>/result.json`.

## Cách tính mức cảnh báo

1. Chạy model 5 lần mỗi giây video (`INFER_FPS`) với conf 0.5, rồi làm mượt theo thời gian để mask không giật:
   - **flood/road**: lấy trung bình trượt của mask qua các lần chạy, có ngưỡng trễ (bật khi > 0.5, chỉ tắt khi < 0.25). Một vùng chỉ hiện nhầm trong 1 lần chạy sẽ không hiện ra; một vùng thật bị sót 1 lần vẫn được giữ.
   - **xe**: nối xe giữa các lần chạy theo IoU box. Level là level xuất hiện nhiều nhất trong 7 lần gần nhất (hòa thì lấy level cao hơn). Xe bị sót 1 lần vẫn được giữ lại.
   - **khung hình ở giữa 2 lần chạy**: mask flood/road chuyển dần từ lần chạy trước sang lần chạy sau, thay vì đứng yên rồi nhảy.
2. Với mỗi khung hình được chạy model:
   - `flood_ratio` = diện tích flood / (flood + road). Khi thấy rất ít đường (< 5% khung hình):
     - nếu nước chiếm ≥ 10% khung hình thì coi như đường đã bị nước phủ kín, vẫn chia cho (flood + road), ra gần 100%;
     - nếu nước ít hơn thì chỉ là vũng nước (model sót đường), chia cho cả khung hình để vũng nhỏ không thành 100%.
   - `max_car_level` = level cao nhất của xe (car_L0…car_L4; level càng cao thì nước càng sâu).
   - Khung hình đạt một mức nếu thỏa **một trong hai** điều kiện:

   | Mức | Tên | flood_ratio ≥ | hoặc xe ≥ |
   |---|---|---|---|
   | 0 | An toàn | – | – |
   | 1 | Chú ý | 0.05 | L1 |
   | 2 | Cảnh báo | 0.30 | L2 |
   | 3 | Nguy hiểm | 0.60 | L3 |

3. Chia video thành từng **đoạn 2 giây** (`SEGMENT_SECONDS`). Mức của một đoạn là mức cao nhất mà ít nhất 50% khung hình trong đoạn đạt được. Cách này giúp vài khung hình nhận nhầm không làm mức nhảy lung tung.
4. **Giữ mức** (`LEVEL_HOLD_SECONDS`, mặc định 6 giây): nước không rút trong vài giây, nên mức tăng thì đổi ngay, còn muốn giảm thì phải thấp hơn liên tục 6 giây. Cách này lấp những đoạn model sót nước vì bị người/xe che hoặc nhận nhầm. Khi đang giữ mức, thanh cảnh báo ghi "Giữ mức do vừa ngập vài giây trước". Mỗi đoạn trong `segments` có `raw_level` là mức trước khi giữ.
   - Chỉ giữ mức **trong cùng một cảnh**. Video tin tức hay chuyển sang logo, trường quay, đồ họa. Hệ thống phát hiện chuyển cảnh bằng histogram màu (so với khung hình khoảng 1 giây trước, ngưỡng `SCENE_CUT`). Sang cảnh mới thì không kéo mức cũ sang, và cũng xóa lịch sử làm mượt mask/xe để mask cảnh cũ không đè lên cảnh mới. Thời điểm chuyển cảnh nằm trong `scene_cuts` của kết quả.
   - Ngưỡng đặt cao để người/xe đi ngang che camera hay camera rung không bị coi là chuyển cảnh. Vì vậy cắt từ cảnh đường ngập này sang cảnh đường ngập khác (màu gần giống nhau) thường không bị tính là chuyển cảnh; khi đó vẫn giữ mức, thiên về an toàn. Đặt `LEVEL_HOLD_SECONDS=0` để tắt hẳn việc giữ mức.
5. Các đoạn liền nhau có cùng mức được gộp thành một **giai đoạn**.
6. Mức của cả video là mức cao nhất trong các giai đoạn.

Ngưỡng nằm trong `LEVELS` của `analyzer.py`, sửa theo thực tế. Các tham số khác chỉnh qua biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODEL_PATH` | `weights/best.pt` | đường dẫn model |
| `CONF` | 0.5 | confidence tối thiểu |
| `INFER_FPS` | 5 | số lần chạy model mỗi giây video (giảm nếu máy chậm; có GPU thì đặt bằng fps của video để mượt nhất) |
| `ROAD_MIN_FRAC` | 0.05 | đường chiếm ít nhất bao nhiêu khung hình mới coi là thấy đường |
| `FLOOD_COVER_MIN_FRAC` | 0.10 | không thấy đường mà nước chiếm ít nhất bao nhiêu khung hình thì coi là đường bị phủ kín |
| `LEVEL_HOLD_SECONDS` | 6 | mức chỉ được giảm khi đã thấp hơn liên tục bấy nhiêu giây (0 = tắt) |
| `SCENE_CUT` | 0.6 | khác biệt màu (0–1) so với khoảng 1 giây trước vượt ngưỡng này thì coi là chuyển cảnh (1 = tắt) |
| `MASK_SMOOTH` | 0.6 | mức làm mượt mask flood/road (0 = tắt, càng lớn càng mượt nhưng trễ hơn) |
| `CAR_HISTORY` | 7 | số lần chạy gần nhất dùng để chốt level xe |
| `SEGMENT_SECONDS` | 2 | độ dài mỗi đoạn tính mức |
| `MIN_FRAME_FRAC` | 0.5 | tỉ lệ khung hình tối thiểu để một đoạn đạt mức |
| `OUT_WIDTH` | 960 | chiều rộng video kết quả |
| `MAX_SECONDS` | 600 | chỉ xử lý 10 phút đầu |
| `MAX_UPLOAD_MB` | 500 | dung lượng upload tối đa |
| `FONT_PATH` | tự tìm | font để vẽ chữ tiếng Việt lên video |

## Ví dụ kết quả (`GET /jobs/<job_id>` → `result`)

```json
{
  "warning_level": 3,
  "warning": "Nguy hiểm",
  "duration_s": 10.0,
  "seconds_per_level": {"An toàn": 0, "Chú ý": 4.0, "Cảnh báo": 2.0, "Nguy hiểm": 4.0},
  "flood_ratio_max": 0.885,
  "car_level_max": 3,
  "phases": [
    {"start_s": 0.0, "end_s": 4.0, "level": 1, "warning": "Chú ý", "flood_ratio_max": 0.382, "car_level_max": null},
    {"start_s": 4.0, "end_s": 6.0, "level": 2, "warning": "Cảnh báo", "flood_ratio_max": 0.562, "car_level_max": null},
    {"start_s": 6.0, "end_s": 10.0, "level": 3, "warning": "Nguy hiểm", "flood_ratio_max": 0.885, "car_level_max": 3}
  ]
}
```

## Lưu ý

- Job lưu trong bộ nhớ, tắt server là mất danh sách job. File kết quả vẫn còn trong `backend/outputs/`; xóa thủ công khi đầy ổ.
- Video được xử lý lần lượt từng cái.
- Chạy bằng CPU thì mất khoảng 1–2 lần thời lượng video. Có GPU thì ultralytics tự dùng.
