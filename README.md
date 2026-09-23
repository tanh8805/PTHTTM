# PTHTTM — Phát hiện ngập úng từ video

Web app FastAPI dùng model YOLO segmentation (`best.pt`, class 0 = FLOOD) để tô vùng ngập trong video và đưa ra mức cảnh báo 0–3.

## Cài đặt (Fedora / Linux)

> **Lưu ý:** cần cài git-lfs trước khi clone thì mới tải được `best.pt` (nếu không, `best.pt` chỉ là file pointer vài trăm byte; khi đó chạy `git lfs install && git lfs pull`).

```bash
sudo dnf install git-lfs
git clone https://github.com/tanh8805/PTHTTM && cd PTHTTM
git lfs install && git lfs pull          # tải best.pt thật (~6 MB)

# venv đặt NGOÀI ổ NTFS để tránh lỗi symlink/permission
uv venv -p 3.12 ~/.venvs/pthttm
VIRTUAL_ENV=~/.venvs/pthttm uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=~/.venvs/pthttm uv pip install -r requirements.txt
```

Không có GPU NVIDIA: bỏ dòng cài torch CUDA, `requirements.txt` sẽ cài torch bản thường và app tự chạy CPU.

**Venv:** `~/.venvs/pthttm`

## Chạy

```bash
~/.venvs/pthttm/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Mở http://localhost:8000

Video kết quả được transcode sang H.264 (yuv420p, faststart) bằng ffmpeg đi kèm gói `imageio-ffmpeg`, nên trình duyệt phát được mà không cần ffmpeg hệ thống.
