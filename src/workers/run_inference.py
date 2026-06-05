from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.identity import one_hot
from src.core.job_store import utc_now
from src.core.numpy_compat import patch_numpy_legacy_aliases
from src.core.settings import PROJECT_ROOT, load_demo_settings, load_model_config, resolve_path
from src.inference.flame_runner import flame_vertices
from src.preprocessing.audio import load_audio_for_model
from src.preprocessing.shape import build_shape_sequence
from src.rendering.mesh_renderer import render_vertices_to_video
from src.rendering.video import mux_audio, transcode_video

MODEL_CACHE: dict[tuple[Any, ...], torch.nn.Module] = {}
FLAME_CACHE: dict[tuple[Any, ...], torch.nn.Module] = {}


def update_job(path: Path, **changes: Any) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8"))
    job.update(changes)
    job["updated_at"] = utc_now()
    path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    return job


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} 缺失：{path}")


@dataclass
class FlameConfig:
    flame_model_path: str
    static_landmark_embedding_path: str
    dynamic_landmark_embedding_path: str
    shape_params: int = 300
    expression_params: int = 100
    batch_size: int = 256
    use_face_contour: bool = True
    use_3D_translation: bool = True


def configure_torch() -> None:
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except Exception:
        return torch.load(str(path), map_location=device)


def log_step(started_at: float, label: str) -> float:
    now = time.perf_counter()
    print(f"[timing] {label}: {now - started_at:.2f}s", flush=True)
    return now


def reset_third_party_modules() -> None:
    prefixes = ("FLAME", "VQVAE2", "Generation", "Diffusion", "Utils")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            del sys.modules[name]


def add_third_party_path(model_key: str) -> Path:
    # 两个模型源码里有同名包，切换模型前先清理旧模块，避免 import 到上一次的实现。
    reset_third_party_modules()
    folder = "dectalk3d" if model_key == "dectalk" else "prodectalk3d"
    path = PROJECT_ROOT / "third_party" / folder
    for existing in [
        str(PROJECT_ROOT / "third_party" / "dectalk3d"),
        str(PROJECT_ROOT / "third_party" / "prodectalk3d"),
    ]:
        while existing in sys.path:
            sys.path.remove(existing)
    sys.path.insert(0, str(path))
    return path


def load_flame(settings, device: torch.device, batch_size: int, model_key: str):
    cache_key = ("flame", model_key, str(device), batch_size)
    cached = FLAME_CACHE.get(cache_key)
    if cached is not None:
        print(f"[cache] reuse FLAME batch_size={batch_size}", flush=True)
        return cached

    # FLAME 依赖 chumpy 旧接口，导入前补上 numpy 1.24 删除的别名。
    patch_numpy_legacy_aliases()
    from FLAME.FLAME import FLAME

    # Avoid importing Utils.py here; it pulls open_clip/torchvision before the
    # model needs them and can add a long delay to FLAME-only setup.
    config = FlameConfig(
        flame_model_path=str(settings.flame_dir / "flame_model" / "generic_model.pkl"),
        static_landmark_embedding_path=str(settings.flame_dir / "flame_model" / "flame_static_embedding.pkl"),
        dynamic_landmark_embedding_path=str(settings.flame_dir / "flame_model" / "flame_dynamic_embedding.npy"),
        batch_size=batch_size,
    )
    flame_model = FLAME(
        config
    )
    flame_model = flame_model.to(device).eval()
    FLAME_CACHE[cache_key] = flame_model
    return flame_model


def load_dectalk_model(config: dict[str, Any], device: torch.device):
    # DecTalk3D 第二阶段权重 generation.pth 已包含 VQ-VAE 子模块参数。
    weights = config["weights"]
    generation_path = resolve_path(weights["generation"])
    require_file(generation_path, "DecTalk3D generation 权重")
    cache_key = (
        "dectalk",
        str(device),
        str(generation_path),
        generation_path.stat().st_mtime_ns,
    )
    cached = MODEL_CACHE.get(cache_key)
    if cached is not None:
        print("[cache] reuse DecTalk3D model", flush=True)
        return cached

    from Generation.FaceGeneration import FaceGenerationModel

    state = load_checkpoint(generation_path, device)
    state_dict = state.get("model_state_dict", state)

    stage1 = config["stage1"]
    stage2 = config["stage2"]
    model = FaceGenerationModel(
        stage1["embed_dim"],
        stage1["num_heads"],
        stage1["num_layers_top"],
        stage1["num_layers_bottom"],
        stage1["num_layers_decoder"],
        stage1["num_embeddings_top"],
        stage1["num_embeddings_bottom"],
        stage2["num_heads"],
        stage2["num_layers_top"],
        stage2["num_layers_bottom"],
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    MODEL_CACHE[cache_key] = model
    return model


def run_dectalk(config: dict[str, Any], person, text, audio, device):
    model = load_dectalk_model(config, device)
    with torch.inference_mode():
        return model.predict(person, [text], audio)


def load_prodectalk_model(config: dict[str, Any], device: torch.device):
    # ProDecTalk3D 第二阶段权重 diffusion.pth 已包含 VQ-VAE 子模块参数。
    weights = config["weights"]
    diffusion_path = resolve_path(weights["diffusion"])
    require_file(diffusion_path, "ProDecTalk3D diffusion 权重")
    cache_key = (
        "prodectalk",
        str(device),
        str(diffusion_path),
        diffusion_path.stat().st_mtime_ns,
    )
    cached = MODEL_CACHE.get(cache_key)
    if cached is not None:
        print("[cache] reuse ProDecTalk3D model", flush=True)
        return cached

    from Diffusion.Diffusion import FaceGenerationModel
    from Utils import EMA

    state = load_checkpoint(diffusion_path, device)
    state_dict = state.get("model_state_dict", state)

    stage1 = config["stage1"]
    stage2 = config["stage2"]
    gpu = int(config.get("gpu", 0))
    model = FaceGenerationModel(
        stage1["embed_dim"],
        stage1["num_heads"],
        stage1["num_layers_style"],
        stage1["num_layers_top"],
        stage1["num_layers_bottom"],
        stage1["num_embeddings"],
        stage2["num_heads"],
        stage2["num_layers_temporal"],
        stage2["num_layers_semantic"],
        stage2["num_layers"],
        gpu,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    if "ema_state_dict" in state:
        # 原 ProDecTalk3D 预测脚本使用 EMA 权重，这里保持一致。
        ema = EMA(model)
        ema.shadow = state["ema_state_dict"]
        ema.apply_shadow(model)
    model.eval()
    MODEL_CACHE[cache_key] = model
    return model


def run_prodectalk(config: dict[str, Any], person, text, audio, device):
    model = load_prodectalk_model(config, device)
    sample = config.get("sample", {})
    with torch.inference_mode():
        return model.sample(
            person,
            [text],
            audio,
            num_sampling_steps_top=sample.get("num_sampling_steps_top", 25),
            num_sampling_steps_bottom=sample.get("num_sampling_steps_bottom", 25),
            temperature=sample.get("temperature", 0.2),
            k=sample.get("top_k", 5),
        )


def selected_shape_id(job: dict[str, Any]) -> str:
    return job.get("shape_id") or job["person_id"]


def run(job_file: Path) -> None:
    configure_torch()
    step_started_at = time.perf_counter()
    settings = load_demo_settings()
    # 第三方源码首次加载 HuBERT/OpenCLIP 时会查缓存，固定到项目目录便于离线复用。
    torchaudio_cache = PROJECT_ROOT / "assets" / "pretrained" / "torchaudio"
    openclip_cache = PROJECT_ROOT / "assets" / "pretrained" / "clip"
    torchaudio_cache.mkdir(parents=True, exist_ok=True)
    openclip_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHAUDIO_MODEL_DIR", str(torchaudio_cache))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(openclip_cache))

    job = update_job(job_file, status="running", error=None)
    try:
        model_key = job["model"]
        config = load_model_config(model_key)
        add_third_party_path(model_key)
        step_started_at = log_step(step_started_at, "load config")

        # 优先使用配置中的 GPU；没有 CUDA 时自动回退 CPU，方便先验证 WebUI 流程。
        gpu = int(config.get("gpu", 0))
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        print(f"[runtime] device={device}", flush=True)
        audio, valid_frames = load_audio_for_model(
            Path(job["audio_path"]),
            target_sample_rate=settings.sample_rate,
            max_seconds=settings.max_audio_seconds,
            pad_to_max=True,
            frame_alignment=16,
        )
        print(
            f"[runtime] audio_frames={int(audio.shape[1]) // 1920}, valid_frames={valid_frames}",
            flush=True,
        )
        step_started_at = log_step(step_started_at, "load audio")
        person = torch.tensor([one_hot(job["person_id"])], dtype=torch.float32)

        # shape 只参与 FLAME 顶点解码，不参与 DecTalk3D / ProDecTalk3D 的身份条件。
        shape_id = selected_shape_id(job)
        audio = audio.to(device)
        person = person.to(device)
        step_started_at = log_step(step_started_at, "prepare tensors")

        if model_key == "dectalk":
            update_job(job_file, progress="模型推理")
            exp, jaw = run_dectalk(config, person, job["text"], audio, device)
        else:
            update_job(job_file, progress="扩散采样")
            exp, jaw = run_prodectalk(config, person, job["text"], audio, device)
        print(f"[runtime] generated_frames={int(exp.shape[1])}", flush=True)
        step_started_at = log_step(step_started_at, "model inference")

        exp = exp.squeeze(0)
        jaw = jaw.squeeze(0)
        # 两个模型最终都输出 FLAME 表情和下颌参数，再统一解码为逐帧顶点。
        update_job(job_file, progress="FLAME 解码")
        shape = build_shape_sequence(
            settings.mean_shape_dir,
            shape_id,
            frames=int(exp.shape[0]),
        ).to(device)
        flame_model = load_flame(settings, device, batch_size=int(exp.shape[0]), model_key=model_key)
        vertices = flame_vertices(flame_model, shape, exp, jaw)
        step_started_at = log_step(step_started_at, "flame decode")
        keep_frames = min(valid_frames, int(vertices.shape[0]))
        vertices_np = vertices[:keep_frames].detach().cpu().numpy()
        exp_np = exp[:keep_frames].detach().cpu().numpy()
        jaw_np = jaw[:keep_frames].detach().cpu().numpy()
        shape_np = shape[:keep_frames].detach().cpu().numpy()

        output_dir = Path(job["output_dir"])
        vertices_path = output_dir / "vertices.npy"
        drive_params_path = output_dir / "drive_params.npz"
        silent_video_path = output_dir / "mesh_silent.mp4"
        final_video_path = output_dir / "mesh.mp4"

        np.save(vertices_path, vertices_np)
        # drive_params.npz 是后续接 3DGS 渲染的稳定数据出口。
        np.savez(
            drive_params_path,
            vertices=vertices_np,
            exp=exp_np,
            jaw=jaw_np,
            shape=shape_np,
            valid_len=keep_frames,
            person_id=job["person_id"],
            shape_id=shape_id,
            text=job["text"],
            model=model_key,
        )
        step_started_at = log_step(step_started_at, "save outputs")
        update_job(job_file, progress="渲染预览视频")
        render_vertices_to_video(
            vertices_np,
            settings.flame_dir / "flame_sample.ply",
            silent_video_path,
            fps=settings.fps,
            width=settings.frame_width,
            height=settings.frame_height,
        )
        step_started_at = log_step(step_started_at, "render video")
        update_job(job_file, progress="合成音频")
        try:
            mux_audio(
                silent_video_path,
                Path(job["audio_path"]),
                final_video_path,
                preset=settings.video_preset,
                crf=settings.video_crf,
            )
        except Exception:
            try:
                transcode_video(
                    silent_video_path,
                    final_video_path,
                    preset=settings.video_preset,
                    crf=settings.video_crf,
                )
            except Exception:
                shutil.copyfile(silent_video_path, final_video_path)
        step_started_at = log_step(step_started_at, "mux audio")

        update_job(
            job_file,
            status="completed",
            video_path=str(final_video_path),
            vertices_path=str(vertices_path),
            drive_params_path=str(drive_params_path),
        )
    except Exception as exc:
        update_job(job_file, status="failed", error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", required=True, type=Path)
    args = parser.parse_args()
    run(args.job_file)


if __name__ == "__main__":
    main()
