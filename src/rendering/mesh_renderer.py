from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh


def render_vertices_to_video(
    vertices: np.ndarray,
    template_mesh_path: Path,
    output_path: Path,
    fps: int = 25,
    width: int = 960,
    height: int = 760,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 渲染参数与 DecTalk3D 的 Render0.py、ProDecTalk3D 的 Render.py 保持一致。
    ref_mesh = trimesh.load_mesh(str(template_mesh_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 20, aspectRatio=1.414)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    camera_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        for frame_vertices in vertices:
            ref_mesh.vertices = frame_vertices
            scene = pyrender.Scene()
            # 原始渲染脚本使用 pyrender 默认平滑效果，因此这里不显式传 smooth=False。
            scene.add(pyrender.Mesh.from_trimesh(ref_mesh))
            scene.add(camera, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            color, _ = renderer.render(scene)
            writer.write(color)
    finally:
        writer.release()
        renderer.delete()
    return output_path
