from __future__ import annotations

import torch


def flame_vertices(flame_model, shape: torch.Tensor, exp: torch.Tensor, jaw: torch.Tensor) -> torch.Tensor:
    frames = int(exp.shape[0])
    shape = shape[:frames].to(exp.device)
    # FLAME 的 pose 前 3 维是头部全局旋转；本 Demo 只驱动下颌，因此全局旋转置零。
    pose_params = torch.cat((torch.zeros((frames, 3), device=exp.device), jaw[:frames]), dim=1)
    vertices, _ = flame_model(shape, exp[:frames], pose_params)
    return vertices
