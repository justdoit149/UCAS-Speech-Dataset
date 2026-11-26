"""模型清单、完整性检查与按需下载。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    repo: str
    subdir: str
    files: tuple[str, ...]


MODEL_SPECS: dict[str, ModelSpec] = {
    "vad": ModelSpec(
        "modelscope", "xukaituo/FireRedVAD", "FireRedVAD",
        ("VAD/cmvn.ark", "VAD/model.pth.tar",
         "AED/cmvn.ark", "AED/model.pth.tar"),
    ),
    "sv": ModelSpec(
        "modelscope", "iic/speech_campplus_sv_zh-cn_16k-common", "campplus",
        ("configuration.json", "campplus_cn_common.bin"),
    ),
    "asr06b": ModelSpec(
        "modelscope", "Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR-0.6B",
        ("config.json", "model.safetensors", "tokenizer_config.json",
         "vocab.json", "merges.txt", "preprocessor_config.json"),
    ),
    "asr17b": ModelSpec(
        "modelscope", "Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B",
        ("config.json", "model.safetensors", "tokenizer_config.json",
         "vocab.json", "merges.txt", "preprocessor_config.json"),
    ),
    "noise": ModelSpec(
        "huggingface", "Etherll/NoisySpeechDetection-v0.2", "NoisySpeechDetection",
        ("config.json", "preprocessor_config.json", "model.safetensors"),
    ),
}

# 模型 key 对应配置中的 section；vad/AED 共用 FireRedVAD 目录。
_SECTION = {"vad": "vad", "sv": "sv", "asr06b": "asr", "asr17b": "asr", "noise": "noise"}


def model_dest(cfg, key: str) -> Path:
    """返回某个模型在本地应所在的目录。"""
    return cfg.model_dir(_SECTION[key])


def missing_files(cfg, key: str) -> list[str]:
    spec = MODEL_SPECS[key]
    dest = model_dest(cfg, key)
    return [name for name in spec.files if not (dest / name).exists()]


def required_model_keys(cfg, stages: Iterable[str]) -> list[str]:
    """按将要运行的阶段推导需要哪些模型。"""
    if bool(cfg.get("run.fake_models", False)):
        return []
    stages = set(stages)
    keys: list[str] = []
    if "vad" in stages:
        keys.append("vad")
    if "sv" in stages:
        keys.append("sv")
    if "asr" in stages:
        sub = str(cfg.get("asr.model_subdir", "")).lower()
        if "1.7b" in sub:
            keys.append("asr17b")
        elif "0.6b" in sub:
            keys.append("asr06b")
        else:
            logger.warning(
                "无法从 asr.model_subdir=%r 判断模型规模，跳过自动下载；"
                "如果本地模型缺失，ASR 阶段会直接报错",
                cfg.get("asr.model_subdir"),
            )
    if "noise" in stages:
        keys.append("noise")
    return keys


def download_spec(spec: ModelSpec, dest: Path, hf_endpoint: str | None = None) -> None:
    """按模型规格下载到指定目录，不检查配置。"""
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("下载模型：%s -> %s", spec.repo, dest)
    try:
        if spec.backend == "modelscope":
            from modelscope import snapshot_download
            snapshot_download(spec.repo, local_dir=str(dest))
        elif spec.backend == "huggingface":
            # 必须在导入 huggingface_hub 之前设置，镜像才会生效
            os.environ.setdefault(
                "HF_ENDPOINT", hf_endpoint or "https://hf-mirror.com")
            os.environ.setdefault("HF_HUB_ENABLE_SYMLINKS", "0")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=spec.repo, local_dir=str(dest))
        else:
            raise ModelError(f"未知模型后端：{spec.backend}")
    except Exception as exc:
        raise ModelError(f"下载失败（{spec.repo}）：{exc}") from exc


def download_model(cfg, key: str) -> Path:
    """下载一个模型到配置指定的目录，并检查关键文件是否齐全。"""
    spec = MODEL_SPECS[key]
    dest = model_dest(cfg, key)
    download_spec(spec, dest)
    missing = missing_files(cfg, key)
    if missing:
        raise ModelError(
            f"下载后仍缺少文件：{key} -> {dest}：{', '.join(missing)}")
    return dest


def ensure_models(
    cfg,
    stages: Iterable[str],
    downloader: Callable[[object, str], object] | None = None,
) -> None:
    """确保所选阶段需要的模型齐全；缺失时自动下载。"""
    keys = required_model_keys(cfg, stages)
    if not keys:
        return
    download = downloader or download_model
    for key in keys:
        dest = model_dest(cfg, key)
        missing = missing_files(cfg, key)
        if not missing:
            logger.info("模型齐全：%s", dest)
            continue
        logger.warning("模型缺失（%s），开始自动下载：%s", ", ".join(missing), dest)
        try:
            download(cfg, key)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"自动下载模型 {key} 失败（{dest}）：{exc}") from exc
        missing = missing_files(cfg, key)
        if missing:
            raise ModelError(
                f"模型 {key} 仍不完整：{dest}；缺少 {', '.join(missing)}")
        logger.info("模型下载完成：%s", dest)
