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


def shape_candidates(mean_shape_dir: Path, shape_id: str) -> list[Path]:
    """返回某个身份可能对应的平均 shape 文件路径。"""
    return [
        mean_shape_dir / f"{shape_id}_mean_shape.npz",
        mean_shape_dir / f"{shape_id}_mean_shape.npy",
        mean_shape_dir / f"{shape_id}.npz",
        mean_shape_dir / f"{shape_id}.npy",
    ]


def list_shape_ids(mean_shape_dir: Path) -> list[str]:
    """扫描 dataset/mean_shape，提取可用于下拉框的 shape 身份。"""
    if not mean_shape_dir.exists():
        return []

    ids: set[str] = set()
    for path in mean_shape_dir.iterdir():
        if path.suffix not in {".npz", ".npy"}:
            continue
        stem = path.stem
        ids.add(stem[: -len("_mean_shape")] if stem.endswith("_mean_shape") else stem)
    return sorted(ids)


def shape_exists(mean_shape_dir: Path, shape_id: str) -> bool:
    if shape_id == "zero_shape":
        return True
    return any(path.exists() for path in shape_candidates(mean_shape_dir, shape_id))


def load_shape_vector(mean_shape_dir: Path, shape_id: str) -> np.ndarray:
    if shape_id == "zero_shape":
        return np.zeros((300,), dtype=np.float32)

    # shape 身份只控制 FLAME 脸型，可以独立于 MEAD 身份向量选择。
    for path in shape_candidates(mean_shape_dir, shape_id):
        if path.exists():
            array = _load_npz_shape(path) if path.suffix == ".npz" else np.load(path)
            array = np.asarray(array, dtype=np.float32)
            if array.ndim == 2:
                array = array[0]
            if array.shape[0] < 300:
                raise ValueError(f"Mean shape for {shape_id} has fewer than 300 values")
            return array[:300]
    raise FileNotFoundError(f"Mean shape not found for {shape_id}")


def build_shape_sequence(mean_shape_dir: Path, shape_id: str, frames: int = 256) -> torch.Tensor:
    shape = load_shape_vector(mean_shape_dir, shape_id)
    sequence = np.repeat(shape[None, :], frames, axis=0)
    return torch.from_numpy(sequence).float()
