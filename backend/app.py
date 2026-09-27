"""Backend đơn giản: upload video -> trả về mức cảnh báo ngập.

Chạy:  uvicorn app:app --host 0.0.0.0 --port 8000
Mở http://localhost:8000 để upload bằng trình duyệt, hoặc xem API ở http://localhost:8000/docs
"""
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

import analyzer

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

app = FastAPI(title="Cảnh báo ngập từ video")


@app.on_event("startup")
def load_model():
    analyzer.get_model()  # nạp model 1 lần khi khởi động, báo lỗi sớm nếu thiếu best.pt


@app.get("/health")
def health():
    return {"status": "ok", "model": analyzer.MODEL_PATH}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), include_frames: bool = False):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Chỉ nhận video {sorted(VIDEO_EXTS)}")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        if os.path.getsize(path) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"Video quá {MAX_UPLOAD_MB}MB")
        result = await run_in_threadpool(analyzer.analyze_video, path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        os.remove(path)
    if not include_frames:
        result.pop("frames")
    return {"filename": file.filename, **result}


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><meta charset="utf-8"><title>Cảnh báo ngập</title>
<body style="font-family:sans-serif;max-width:640px;margin:40px auto;padding:0 16px">
<h2>Upload video camera đường phố</h2>
<input type="file" id="f" accept="video/*"> <button onclick="go()">Phân tích</button>
<h1 id="lv"></h1><pre id="out"></pre>
<script>
const colors=["green","goldenrod","darkorange","red"];
async function go(){
  const f=document.getElementById('f').files[0]; if(!f) return;
  const fd=new FormData(); fd.append('file',f);
  lv.textContent='Đang phân tích...'; lv.style.color='gray'; out.textContent='';
  const r=await fetch('/analyze',{method:'POST',body:fd}); const j=await r.json();
  if(!r.ok){lv.textContent='Lỗi'; lv.style.color='red'; out.textContent=j.detail; return;}
  lv.textContent='Mức '+j.warning_level+': '+j.warning; lv.style.color=colors[j.warning_level];
  out.textContent=JSON.stringify(j,null,2);
}
</script></body>"""
