from __future__ import annotations

from pathlib import Path

import shutil
import subprocess


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    raise FileNotFoundError("未找到 ffmpeg，请先安装 FFmpeg 并将其加入 PATH。")


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 执行失败：{detail}")


def transcode_video(video_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV 写出的 mp4v 在浏览器里可能黑屏，统一转成 H.264/yuv420p。
    _run_ffmpeg(
        [
            _find_ffmpeg(),
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return output_path


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 合成时同时重编码视频，保证最终 mesh.mp4 能被浏览器稳定播放。
    _run_ffmpeg(
        [
            _find_ffmpeg(),
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return output_path
