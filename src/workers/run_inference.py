from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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


def update_job(path: Path, **changes: Any) -> dict[str, Any]:
    job = json.loads(path.read_text(encoding="utf-8"))
    job.update(changes)
    job["updated_at"] = utc_now()
    path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    return job


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} 缺失：{path}")


def reset_third_party_modules() -> None:
    prefixes = ("FLAME", "VQVAE2", "Generation", "Diffusion", "Utils")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            del sys.modules[name]


def add_third_party_path(model_key: str) -> Path:
    # 两个原项目有同名包，切换模型前先清掉旧模块，避免 import 到上一次的实现。
    reset_third_party_modules()
    folder = "dectalk3d" if model_key == "dectalk" else "prodectalk3d"
    path = PROJECT_ROOT / "third_party" / folder
    sys.path.insert(0, str(path))
    return path


def load_flame(settings, device: torch.device):
    # FLAME 依赖 chumpy 旧接口，先补 numpy 1.24 删除的别名再导入。
    patch_numpy_legacy_aliases()
    from FLAME.FLAME import FLAME
    from Utils import Config

    # FLAME 类来自当前模型的 third_party/FLAME，实际模型文件统一读取 configs/demo.yaml 的 flame_dir。
    flame_model = FLAME(
        Config(
            300,
            100,
            str(settings.flame_dir / "flame_model" / "generic_model.pkl"),
            str(settings.flame_dir / "flame_model" / "flame_static_embedding.pkl"),
            str(settings.flame_dir / "flame_model" / "flame_dynamic_embedding.npy"),
        )
    )
    return flame_model.to(device).eval()


def run_dectalk(config: dict[str, Any], person, text, audio, device):
    # DecTalk3D 第二阶段是条件生成模型，对应原仓库 Generation/FaceGeneration.py。
    from Generation.FaceGeneration import FaceGenerationModel

    weights = config["weights"]
    vqvae_path = resolve_path(weights["vqvae"])
    generation_path = resolve_path(weights["generation"])
    require_file(vqvae_path, "DecTalk3D VQ-VAE 权重")
    require_file(generation_path, "DecTalk3D generation 权重")

    stage1 = config["stage1"]
    stage2 = config["stage2"]
    model = FaceGenerationModel(
        str(vqvae_path),
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
    state = torch.load(str(generation_path), map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    with torch.no_grad():
        return model.predict(person, [text], audio)


def run_prodectalk(config: dict[str, Any], person, text, audio, device):
    # ProDecTalk3D 第二阶段是向量量化扩散模型，对应原仓库 Diffusion/Diffusion.py。
    from Diffusion.Diffusion import FaceGenerationModel
    from Utils import EMA

    weights = config["weights"]
    vqvae_path = resolve_path(weights["vqvae"])
    diffusion_path = resolve_path(weights["diffusion"])
    require_file(vqvae_path, "ProDecTalk3D VQ-VAE 权重")
    require_file(diffusion_path, "ProDecTalk3D diffusion 权重")

    stage1 = config["stage1"]
    stage2 = config["stage2"]
    gpu = int(config.get("gpu", 0))
    model = FaceGenerationModel(
        str(vqvae_path),
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
    state = torch.load(str(diffusion_path), map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    if "ema_state_dict" in state:
        # ProDecTalk3D 预测脚本使用 EMA 权重生成，这里保持同样的推理权重。
        ema = EMA(model)
        ema.shadow = state["ema_state_dict"]
        ema.apply_shadow(model)
    model.eval()
    sample = config.get("sample", {})
    with torch.no_grad():
        return model.sample(
            person,
            [text],
            audio,
            num_sampling_steps_top=sample.get("num_sampling_steps_top", 25),
            num_sampling_steps_bottom=sample.get("num_sampling_steps_bottom", 25),
            temperature=sample.get("temperature", 0.2),
            k=sample.get("top_k", 5),
        )


def run(job_file: Path) -> None:
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

        # 按配置选择 GPU；没有 CUDA 时自动退回 CPU，方便先验证 WebUI 流程。
        gpu = int(config.get("gpu", 0))
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        audio, valid_frames = load_audio_for_model(
            Path(job["audio_path"]),
            target_sample_rate=settings.sample_rate,
            max_seconds=settings.max_audio_seconds,
        )
        person = torch.tensor([one_hot(job["person_id"])], dtype=torch.float32)
        # shape 只参与 FLAME 顶点解码；生成模型输出的是 exp 和 jaw。
        shape = build_shape_sequence(
            settings.mean_shape_dir,
            job["person_id"],
            job["shape_mode"],
            frames=int(settings.sample_rate * settings.max_audio_seconds) // 1920,
        )
        audio = audio.to(device)
        person = person.to(device)
        shape = shape.to(device)
        flame_model = load_flame(settings, device)

        if model_key == "dectalk":
            exp, jaw = run_dectalk(config, person, job["text"], audio, device)
        else:
            exp, jaw = run_prodectalk(config, person, job["text"], audio, device)

        exp = exp.squeeze(0)
        jaw = jaw.squeeze(0)
        # 两个模型最终都统一为 FLAME 参数，再由 FLAME 解码为每帧顶点。
        vertices = flame_vertices(flame_model, shape, exp, jaw)
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
            text=job["text"],
            model=model_key,
        )
        render_vertices_to_video(
            vertices_np,
            settings.flame_dir / "flame_sample.ply",
            silent_video_path,
            fps=settings.fps,
            width=settings.frame_width,
            height=settings.frame_height,
        )
        try:
            mux_audio(silent_video_path, Path(job["audio_path"]), final_video_path)
        except Exception:
            try:
                transcode_video(silent_video_path, final_video_path)
            except Exception:
                shutil.copyfile(silent_video_path, final_video_path)

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
