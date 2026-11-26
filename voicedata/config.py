"""配置加载、路径解析与阶段参数哈希。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

STAGES = ("decode", "vad", "sv", "asr", "noise", "export")

# 导出的孤立填充词（去标点后完全相等才算；同字重复 <=4 次也视为填充词）
DEFAULT_FILLER_TEXTS = [
    "嗯", "恩", "啊", "哦", "噢", "喔", "呃", "额", "唉", "诶", "哎", "喂",
    "好", "好的", "好吧", "好嘞", "好滴", "对", "对的", "是", "是的", "行", "OK", "okay",
]


class ConfigError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_hash(params: Dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


class Config:
    def __init__(self, data: Dict[str, Any], path: Path):
        self.data = data
        self.source_path = Path(path)
        self.base_dir = self.source_path.parent

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise ConfigError(f"配置文件不存在：{path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigError("配置文件顶层必须是映射")
        cfg = cls(data, path)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        required = [
            "paths.input_dir",
            "paths.work_root",
            "paths.output_root",
            "paths.model_dir",
            "paths.template_path",
            "run.device",
            "vad.speech_threshold",
            "aed.model_subdir",
            "aed.speech_threshold",
            "sv.threshold",
            "asr.model_subdir",
            "noise.model_subdir",
            "export.schema_version",
            "export.format",
        ]
        missing = [k for k in required if self.get(k) is None]
        if missing:
            raise ConfigError(f"缺少配置项：{', '.join(missing)}")
        fmt = str(self.get("export.format", "zip")).lower()
        if fmt not in ("zip", "json"):
            raise ConfigError(f"export.format 只支持 zip / json，实际为：{fmt}")

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve(self, value: str) -> Path:
        p = Path(value)
        if not p.is_absolute():
            p = self.base_dir / p
        return p.resolve()

    def path(self, dotted: str) -> Path:
        return self.resolve(str(self.get(dotted)))

    def job_dir(self, kind: str, job: str) -> Path:
        root = self.path("paths.work_root" if kind == "work" else "paths.output_root")
        return root / job

    def threshold(self, override: float | None = None) -> float:
        if override is not None:
            return float(override)
        return float(self.get("sv.threshold", 0.4))

    def stage_params(self, stage: str, job: str, threshold: float | None = None) -> Dict[str, Any]:
        if stage == "decode":
            return {
                "version": 1,
                "sample_rate": int(self.get("decode.sample_rate", 16000)),
                "channels": 1,
                "sample_format": "pcm_s16le",
            }
        if stage == "vad":
            return {
                "version": 1,
                "model": self.get("vad.model_name", "FireRedVAD"),
                "sample_rate": int(self.get("decode.sample_rate", 16000)),
                "speech_threshold": float(self.get("vad.speech_threshold", 0.4)),
                "min_speech_ms": int(self.get("vad.min_speech_ms", 350)),
                "min_silence_ms": int(self.get("vad.min_silence_ms", 300)),
                "max_segment_s": float(self.get("vad.max_segment_s", 20.0)),
                "pad_ms": int(self.get("vad.pad_ms", 50)),
                "window_s": float(self.get("vad.window_s", 600)),
                "context_s": float(self.get("vad.context_s", 20)),
                "aed": {
                    "enabled": bool(self.get("aed.enabled", True)),
                    "model": self.get("aed.model_subdir", "FireRedVAD/AED"),
                    "speech_threshold": float(self.get("aed.speech_threshold", 0.4)),
                    "min_speech_ms": int(self.get("aed.min_speech_ms", 300)),
                    "merge_gap_ms": int(self.get("aed.merge_gap_ms", 800)),
                    "pad_ms": int(self.get("aed.pad_ms", 50)),
                    "singing_threshold": float(self.get("aed.singing_threshold", 0.5)),
                    "max_merged_s": float(self.get("aed.max_merged_s", 12.0)),
                },
            }
        if stage == "sv":
            template = self.path("paths.template_path")
            return {
                "version": 1,
                "model": self.get("sv.model_name", "campplus_sv_zh-cn_16k-common"),
                "template_sha256": file_sha256(template) if template.exists() else "missing",
                "normalize": True,
                "score": "max_cosine",
            }
        if stage == "asr":
            return {
                "version": 1,
                "model": self.get("asr.model_name", self.get("asr.model_subdir")),
                "language": self.get("asr.language", "Chinese"),
                "max_new_tokens": int(self.get("asr.max_new_tokens", 256)),
                # 记录 token 平均 logprob（供低置信过滤）；旧结果需要补算一次
                "confidence": True,
            }
        if stage == "noise":
            return {
                "version": 1,
                "model": self.get("noise.model_name", self.get("noise.model_subdir")),
                "threshold": float(self.get("noise.threshold", 0.5)),
            }
        if stage == "export":
            fmt = str(self.get("export.format", "zip")).lower()
            return {
                "version": 1,
                "schema_version": str(self.get("export.schema_version", "2.0")),
                "format": fmt,
                "audio_format": (
                    "flac/16000/1/16bit" if fmt == "zip"
                    else "wav/pcm_s16le/16000/1"
                ),
                "threshold": float(self.threshold(threshold)),
                "asr_min_score": float(self.get("sv.asr_min_score", 0.30)),
                "drop_empty_text": bool(self.get("export.drop_empty_text", True)),
                "include_scores": bool(self.get("export.include_scores", True)),
                "drop_fillers": bool(self.get("export.drop_fillers", True)),
                "filler_max_s": float(self.get("export.filler_max_s", 1.2)),
                "filler_texts": [
                    str(x) for x in self.get("export.filler_texts", DEFAULT_FILLER_TEXTS)
                ],
                "min_asr_conf": (
                    None if self.get("export.min_asr_conf") is None
                    else float(self.get("export.min_asr_conf"))
                ),
                # 上游任何参数变化都必须重导出，否则会误用旧产物
                "upstream": {
                    "vad": self.stage_hash("vad", job),
                    "sv": self.stage_hash("sv", job),
                    "asr": self.stage_hash("asr", job),
                    "noise": self.stage_hash("noise", job),
                },
            }
        raise ConfigError(f"未知阶段：{stage}")

    def stage_hash(self, stage: str, job: str, threshold: float | None = None) -> str:
        return params_hash(self.stage_params(stage, job, threshold))

    def model_dir(self, section: str) -> Path:
        sub = self.get(f"{section}.model_subdir")
        if not sub:
            raise ConfigError(f"缺少 {section}.model_subdir")
        return self.resolve(str(self.path("paths.model_dir") / sub))

    def device(self) -> str:
        dev = str(self.get("run.device", "cpu")).lower()
        if dev == "auto":
            try:
                import torch

                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return dev

    def torch_dtype(self):
        import torch

        name = str(self.get("run.dtype", "auto")).lower()
        if name == "auto":
            return torch.bfloat16 if self.device().startswith("cuda") else torch.float32
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if name not in mapping:
            raise ConfigError(f"不支持的 run.dtype：{name}")
        return mapping[name]

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        for dotted, value in overrides.items():
            if value is None:
                continue
            node = self.data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
