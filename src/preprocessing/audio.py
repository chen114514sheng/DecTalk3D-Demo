from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    raise FileNotFoundError("未找到 ffmpeg，请先安装 FFmpeg 并将其加入 PATH。")


def _convert_to_wav(path: Path, target_sample_rate: int) -> Path:
    wav_path = path.with_name(f"{path.stem}_converted_48k.wav")
    command = [
        _find_ffmpeg(),
        "-y",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg 转换音频失败：{stderr}")
    return wav_path


def _load_wav_with_scipy(path: Path):
    sample_rate, data = wavfile.read(str(path))
    array = np.asarray(data)
    if array.ndim == 1:
        array = array[None, :]
    else:
        array = array.T
    if np.issubdtype(array.dtype, np.integer):
        max_value = np.iinfo(array.dtype).max
        array = array.astype(np.float32) / max_value
    else:
        array = array.astype(np.float32)
    return torch.from_numpy(array), sample_rate


def _load_with_fallback(path: Path, target_sample_rate: int):
    try:
        return torchaudio.load(str(path))
    except Exception as exc:
        # Windows 上 torchaudio 对 m4a/aac 的支持不稳定，先转成 48k 单声道 wav 再读。
        wav_path = _convert_to_wav(path, target_sample_rate)
        try:
            return torchaudio.load(str(wav_path))
        except Exception as second_exc:
            try:
                return _load_wav_with_scipy(wav_path)
            except Exception as third_exc:
                raise RuntimeError(
                    f"音频读取失败：{path}。原始错误：{exc}。"
                    f"转换后 wav 错误：{second_exc}。scipy 读取错误：{third_exc}"
                ) from third_exc


def load_audio_for_model(
    path: Path,
    target_sample_rate: int = 48000,
    max_seconds: float = 10.24,
    pad_to_max: bool = True,
    frame_alignment: int = 16,
) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = _load_with_fallback(path, target_sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    samples_per_frame = 1920
    max_samples = int(target_sample_rate * max_seconds)
    max_frames = max_samples // samples_per_frame
    waveform = waveform[:, :max_samples]
    valid_samples = int(waveform.shape[1])
    valid_frames = max(1, min(max_frames, valid_samples // samples_per_frame))

    if pad_to_max:
        padded_frames = max_frames
    else:
        alignment = max(1, frame_alignment)
        padded_frames = min(
            max_frames,
            max(alignment, ((valid_frames + alignment - 1) // alignment) * alignment),
        )
    padded_samples = padded_frames * samples_per_frame

    waveform = waveform[:, :padded_samples]
    if valid_samples < padded_samples:
        padding = padded_samples - valid_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    audio = waveform.transpose(0, 1).unsqueeze(0)
    # 模型每 1920 个采样点对应一帧表情，valid_frames 用于裁掉补零尾部。
    return audio.float(), valid_frames
