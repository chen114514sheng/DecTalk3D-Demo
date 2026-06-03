from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DemoSettings:
    """演示项目的运行时配置，所有相对路径都会解析到项目根目录下。"""

    root: Path
    runtime_dir: Path
    uploads_dir: Path
    outputs_dir: Path
    jobs_dir: Path
    mean_shape_dir: Path
    flame_dir: Path
    max_audio_seconds: float
    sample_rate: int
    fps: int
    frame_width: int
    frame_height: int


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def load_demo_settings(config_path: Path | None = None) -> DemoSettings:
    path = config_path or PROJECT_ROOT / "configs" / "demo.yaml"
    data = load_yaml(path)
    runtime_dir = resolve_path(data.get("runtime_dir", "runtime"))
    render = data.get("render", {})
    audio = data.get("audio", {})
    settings = DemoSettings(
        root=PROJECT_ROOT,
        runtime_dir=runtime_dir,
        uploads_dir=resolve_path(data.get("uploads_dir", "runtime/uploads")),
        outputs_dir=resolve_path(data.get("outputs_dir", "runtime/outputs")),
        jobs_dir=resolve_path(data.get("jobs_dir", "runtime/jobs")),
        mean_shape_dir=resolve_path(data.get("mean_shape_dir", "dataset/mean_shape")),
        flame_dir=resolve_path(data.get("flame_dir", "assets/flame")),
        max_audio_seconds=float(audio.get("max_seconds", 10.24)),
        sample_rate=int(audio.get("sample_rate", 48000)),
        fps=int(render.get("fps", 25)),
        frame_width=int(render.get("width", 960)),
        frame_height=int(render.get("height", 760)),
    )
    # 启动 API 时提前创建运行目录，避免上传和任务写入时再处理目录缺失。
    for directory in [
        settings.runtime_dir,
        settings.uploads_dir,
        settings.outputs_dir,
        settings.jobs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return settings


def load_model_config(model_key: str) -> dict[str, Any]:
    if model_key not in {"dectalk", "prodectalk"}:
        raise ValueError(f"Unsupported model: {model_key}")
    return load_yaml(PROJECT_ROOT / "configs" / f"{model_key}.yaml")
