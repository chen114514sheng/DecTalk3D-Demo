from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCHAUDIO_CACHE = ROOT / "assets" / "pretrained" / "torchaudio"
OPENCLIP_CACHE = ROOT / "assets" / "pretrained" / "clip"


def main() -> None:
    TORCHAUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    OPENCLIP_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHAUDIO_MODEL_DIR"] = str(TORCHAUDIO_CACHE)
    os.environ["OPENCLIP_CACHE_DIR"] = str(OPENCLIP_CACHE)

    print(f"OpenCLIP 缓存目录：{OPENCLIP_CACHE}")
    print(f"HuBERT 缓存目录：{TORCHAUDIO_CACHE}")

    import open_clip
    import torchaudio

    print("正在检查 OpenCLIP ViT-B-32...")
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        cache_dir=str(OPENCLIP_CACHE),
    )
    print("OpenCLIP 已就绪。")

    print("正在检查 torchaudio HUBERT_BASE...")
    torchaudio.pipelines.HUBERT_BASE.get_model(
        dl_kwargs={"model_dir": str(TORCHAUDIO_CACHE)}
    )
    print("HuBERT 已就绪。")


if __name__ == "__main__":
    main()
