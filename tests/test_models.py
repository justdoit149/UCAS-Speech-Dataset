from __future__ import annotations

from pathlib import Path

import pytest

from voicedata.cli import DEFAULT_CONFIG
from voicedata.models import (
    MODEL_SPECS,
    ModelError,
    ensure_models,
    missing_files,
    model_dest,
    required_model_keys,
)


def _make_complete(cfg, key: str) -> Path:
    dest = model_dest(cfg, key)
    for name in MODEL_SPECS[key].files:
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return dest


def test_default_config_is_server():
    assert DEFAULT_CONFIG == "configs/server.yaml"


def test_required_model_keys_follow_stages(make_config):
    cfg = make_config(**{"run.fake_models": False})
    assert required_model_keys(cfg, ["vad", "sv", "asr", "noise"]) == [
        "vad", "sv", "asr06b", "noise",
    ]
    assert required_model_keys(cfg, ["asr"]) == ["asr06b"]
    assert required_model_keys(cfg, ["vad", "export", "decode"]) == ["vad"]
    assert required_model_keys(cfg, ["decode", "export"]) == []


def test_required_model_keys_pick_asr_size(make_config):
    cfg = make_config(
        **{
            "run.fake_models": False,
            "asr.model_subdir": "Qwen3-ASR-1.7B",
            "asr.model_name": "Qwen3-ASR-1.7B",
        }
    )
    assert required_model_keys(cfg, ["asr"]) == ["asr17b"]


def test_fake_models_need_nothing(make_config):
    cfg = make_config()
    assert required_model_keys(cfg, ["vad", "sv", "asr", "noise"]) == []


def test_ensure_models_skips_complete_and_downloads_missing(make_config):
    cfg = make_config(**{"run.fake_models": False})
    _make_complete(cfg, "vad")
    calls: list[str] = []

    def downloader(c, key):
        calls.append(key)
        _make_complete(c, key)

    ensure_models(cfg, ["vad", "sv", "asr", "noise", "export"], downloader)
    assert calls == ["sv", "asr06b", "noise"]
    assert missing_files(cfg, "vad") == []
    assert missing_files(cfg, "sv") == []
    assert missing_files(cfg, "asr06b") == []
    assert missing_files(cfg, "noise") == []


def test_ensure_models_does_not_download_for_export_or_decode(make_config):
    cfg = make_config(**{"run.fake_models": False})

    def downloader(c, key):  # pragma: no cover - must not be called
        raise AssertionError(f"不应下载 {key}")

    ensure_models(cfg, ["decode", "export"], downloader)


def test_ensure_models_rejects_incomplete_download(make_config):
    cfg = make_config(**{"run.fake_models": False})
    with pytest.raises(ModelError, match="仍不完整"):
        ensure_models(cfg, ["vad"], lambda c, key: None)


def test_ensure_models_wraps_download_error(make_config):
    cfg = make_config(**{"run.fake_models": False})

    def downloader(c, key):
        raise OSError("network down")

    with pytest.raises(ModelError, match="自动下载模型 vad 失败"):
        ensure_models(cfg, ["vad"], downloader)
