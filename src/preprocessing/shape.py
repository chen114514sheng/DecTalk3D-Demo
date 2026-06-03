from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _load_npz_shape(path: Path) -> np.ndarray:
    data = np.load(path)
    if "mean_shape" in data.files:
        return data["mean_shape"]
    if len(data.files) == 1:
        return data[data.files[0]]
    raise ValueError(f"No mean_shape array found in {path}")


def load_shape_vector(mean_shape_dir: Path, person_id: str, mode: str) -> np.ndarray:
    if mode == "zero_shape":
        return np.zeros((300,), dtype=np.float32)
    if mode != "mean_shape":
        raise ValueError(f"Unsupported shape mode: {mode}")

    # 数据集中可能存在 npz 或 npy 两种保存方式，这里按身份依次尝试常见文件名。
    candidates = [
        mean_shape_dir / f"{person_id}_mean_shape.npz",
        mean_shape_dir / f"{person_id}_mean_shape.npy",
        mean_shape_dir / f"{person_id}.npz",
        mean_shape_dir / f"{person_id}.npy",
    ]
    for path in candidates:
        if path.exists():
            array = _load_npz_shape(path) if path.suffix == ".npz" else np.load(path)
            array = np.asarray(array, dtype=np.float32)
            if array.ndim == 2:
                array = array[0]
            if array.shape[0] < 300:
                raise ValueError(f"Mean shape for {person_id} has fewer than 300 values")
            return array[:300]
    raise FileNotFoundError(f"Mean shape not found for {person_id}")


def build_shape_sequence(mean_shape_dir: Path, person_id: str, mode: str, frames: int = 256) -> torch.Tensor:
    shape = load_shape_vector(mean_shape_dir, person_id, mode)
    sequence = np.repeat(shape[None, :], frames, axis=0)
    return torch.from_numpy(sequence).float()
