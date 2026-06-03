from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import DemoSettings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    return uuid.uuid4().hex


def job_dir(settings: DemoSettings, job_id: str) -> Path:
    return settings.jobs_dir / job_id


def job_file(settings: DemoSettings, job_id: str) -> Path:
    return job_dir(settings, job_id) / "job.json"


def write_job(settings: DemoSettings, job_id: str, payload: dict[str, Any]) -> None:
    directory = job_dir(settings, job_id)
    directory.mkdir(parents=True, exist_ok=True)
    job_file(settings, job_id).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_job(settings: DemoSettings, job_id: str) -> dict[str, Any]:
    path = job_file(settings, job_id)
    if not path.exists():
        raise FileNotFoundError(job_id)
    return json.loads(path.read_text(encoding="utf-8"))


def patch_job(settings: DemoSettings, job_id: str, **changes: Any) -> dict[str, Any]:
    payload = read_job(settings, job_id)
    payload.update(changes)
    payload["updated_at"] = utc_now()
    write_job(settings, job_id, payload)
    return payload


def create_job(
    settings: DemoSettings,
    model: str,
    text: str,
    person_id: str,
    shape_id: str,
    audio_path: Path,
) -> dict[str, Any]:
    job_id = new_job_id()
    output_dir = settings.outputs_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "model": model,
        "text": text,
        "person_id": person_id,
        "shape_id": shape_id,
        "audio_path": str(audio_path),
        "output_dir": str(output_dir),
        "video_path": None,
        "vertices_path": None,
        "drive_params_path": None,
        "log_path": str(job_dir(settings, job_id) / "worker.log"),
        "error": None,
    }
    write_job(settings, job_id, payload)
    return payload


def run_worker(settings: DemoSettings, job_id: str) -> None:
    path = job_file(settings, job_id)
    log_path = job_dir(settings, job_id) / "worker.log"
    # 推理过程会加载大模型，放到独立子进程里执行，避免阻塞 FastAPI 主进程。
    command = [
        sys.executable,
        "-m",
        "src.workers.run_inference",
        "--job-file",
        str(path),
    ]
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"Command: {' '.join(command)}\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=str(settings.root),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    payload = read_job(settings, job_id)
    if payload.get("status") == "running":
        # 子进程异常退出但没有写失败状态时，在任务文件里补一条可读错误。
        tail = ""
        if log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        patch_job(
            settings,
            job_id,
            status="failed",
            error=f"Worker exited without completing the job. Return code: {result.returncode}\n{tail}",
        )
