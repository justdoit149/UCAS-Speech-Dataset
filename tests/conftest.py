from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voicedata.config import Config  # noqa: E402


def _merge(dst: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = dst
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


@pytest.fixture
def template_file(tmp_path: Path) -> Path:
    vec = np.zeros(192, dtype=np.float32)
    vec[:96] = 1.0
    vec /= np.linalg.norm(vec)
    p = tmp_path / "说话声纹.json"
    p.write_text(json.dumps([vec.tolist() for _ in range(26)]), encoding="utf-8")
    return p


@pytest.fixture
def make_config(tmp_path: Path, template_file: Path):
    def _make(**overrides) -> Config:
        data = {
            "paths": {
                "input_dir": str(tmp_path / "input"),
                "work_root": str(tmp_path / "work"),
                "output_root": str(tmp_path / "out"),
                "model_dir": str(tmp_path / "models"),
                "template_path": str(template_file),
            },
            "run": {
                "device": "cpu", "dtype": "float32",
                "max_attempts": 1, "retry_backoff_s": 0,
                "fake_models": True,
            },
            "decode": {"sample_rate": 16000},
            "vad": {
                "model_subdir": "FireRedVAD", "model_name": "FireRedVAD",
                "speech_threshold": 0.4, "min_speech_ms": 350,
                "min_silence_ms": 300, "max_segment_s": 20, "pad_ms": 50,
                "window_s": 600, "context_s": 20,
            },
            "aed": {
                "enabled": True, "model_subdir": "FireRedVAD/AED",
                "speech_threshold": 0.4, "min_speech_ms": 300,
                "merge_gap_ms": 800, "pad_ms": 50,
                "singing_threshold": 0.5, "max_merged_s": 12.0,
            },
            "sv": {"model_subdir": "campplus", "model_name": "fake",
                   "threshold": 0.4, "asr_min_score": 0.3, "batch_size": 4},
            "asr": {"model_subdir": "Qwen3-ASR-0.6B", "model_name": "fake",
                    "batch_size": 2, "language": "Chinese", "max_new_tokens": 64},
            "noise": {"model_subdir": "Noise", "model_name": "fake",
                      "batch_size": 2, "threshold": 0.5},
            "export": {"format": "zip", "schema_version": "2.0",
                       "drop_empty_text": True, "include_scores": True,
                       "drop_fillers": True, "filler_max_s": 1.2,
                       "filler_texts": ["嗯", "好的", "OK"], "min_asr_conf": None},
        }
        for k, v in overrides.items():
            _merge(data, k, v)
        cfg = Config(data, tmp_path / "configs" / "test.yaml")
        cfg.validate()
        return cfg
    return _make


@pytest.fixture
def wav_input(tmp_path: Path, make_config) -> tuple[Config, Path]:
    cfg = make_config()
    inp = Path(cfg.get("paths.input_dir"))
    inp.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sr = 16000
    t = np.arange(sr * 6) / sr
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = inp / "直播 测试.wav"
    sf.write(str(p), audio, sr, subtype="PCM_16")
    return cfg, p
