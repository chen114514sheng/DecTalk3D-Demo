from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.core.identity import PERSON_IDS, ensure_identity_file
from src.core.job_store import create_job, read_job, run_worker
from src.core.settings import load_demo_settings

settings = load_demo_settings()
ensure_identity_file(settings.root / "assets" / "identity" / "person_ids.json")
web_dir = settings.root / "web"

app = FastAPI(title="DecTalk3D Demo API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(settings.outputs_dir)), name="outputs")
if web_dir.exists():
    # WebUI 使用 web/ 下的原生前端，避免额外依赖 Node.js。
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")


@app.get("/")
def index():
    index_path = web_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="前端页面不存在")
    # 调试阶段经常改前端文件，首页不缓存可以避免浏览器继续使用旧脚本。
    return FileResponse(index_path, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/options")
def options():
    available_mean_shape = []
    if settings.mean_shape_dir.exists():
        available_mean_shape = sorted(path.name for path in settings.mean_shape_dir.glob("*_mean_shape.npz"))
    # 前端下拉框只依赖这个接口，后续新增模型或 shape 来源时从这里扩展。
    return {
        "models": [
            {"key": "dectalk", "label": "DecTalk3D"},
            {"key": "prodectalk", "label": "ProDecTalk3D"},
        ],
        "person_ids": PERSON_IDS,
        "shape_modes": [
            {"key": "mean_shape", "label": "平均 shape"},
            {"key": "zero_shape", "label": "零 shape"},
        ],
        "mean_shape_files": available_mean_shape,
    }


@app.post("/api/jobs")
def submit_job(
    background_tasks: BackgroundTasks,
    model: str = Form(...),
    text: str = Form(...),
    person_id: str = Form(...),
    shape_mode: str = Form(...),
    audio: UploadFile = File(...),
):
    # 这里先做轻量校验；耗时的模型推理交给后台 worker 子进程。
    if model not in {"dectalk", "prodectalk"}:
        raise HTTPException(status_code=400, detail="不支持的模型")
    if person_id not in PERSON_IDS:
        raise HTTPException(status_code=400, detail="未知的 MEAD 身份")
    if shape_mode not in {"mean_shape", "zero_shape"}:
        raise HTTPException(status_code=400, detail="不支持的 shape 来源")
    if not text.strip():
        raise HTTPException(status_code=400, detail="文本条件不能为空")

    suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    upload_path = settings.uploads_dir / f"{person_id}_{model}_{uuid.uuid4().hex}{suffix}"
    with upload_path.open("wb") as handle:
        shutil.copyfileobj(audio.file, handle)

    # FastAPI BackgroundTasks 只负责启动 worker，任务状态通过 runtime/jobs/{job_id}/job.json 轮询。
    job = create_job(settings, model, text.strip(), person_id, shape_mode, upload_path)
    background_tasks.add_task(run_worker, settings, job["id"])
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = read_job(settings, job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    if job.get("video_path"):
        job["video_url"] = f"/api/jobs/{job_id}/video"
    return job


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    try:
        job = read_job(settings, job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    video_path = job.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="视频尚未生成")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{job_id}.mp4")
