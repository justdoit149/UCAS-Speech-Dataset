"""FireRedVAD 非流式语音活动检测。"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FireRedVADAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        base = cfg.model_dir("vad")
        # 官方仓库布局：<model>/VAD/{cmvn.ark,model.pth.tar}
        self.model_dir = base / "VAD" if (base / "VAD" / "cmvn.ark").exists() else base
        self.use_gpu = cfg.device().startswith("cuda")
        self._vad = None

    def load(self) -> None:
        if self._vad is not None:
            return
        try:
            from fireredvad.vad import FireRedVad, FireRedVadConfig
        except ImportError:
            from fireredvad import FireRedVad, FireRedVadConfig
        if not (self.model_dir / "cmvn.ark").exists():
            raise FileNotFoundError(f"FireRedVAD 模型不完整，缺少 cmvn.ark：{self.model_dir}")
        max_segment_s = float(self.cfg.get("vad.max_segment_s", 20.0))
        config = FireRedVadConfig(
            use_gpu=self.use_gpu,
            smooth_window_size=5,
            speech_threshold=float(self.cfg.get("vad.speech_threshold", 0.4)),
            min_speech_frame=int(float(self.cfg.get("vad.min_speech_ms", 350)) / 10),
            max_speech_frame=int(max_segment_s * 100),
            min_silence_frame=int(float(self.cfg.get("vad.min_silence_ms", 300)) / 10),
            merge_silence_frame=0,
            extend_speech_frame=int(float(self.cfg.get("vad.pad_ms", 50)) / 10),
            chunk_max_frame=30000,
        )
        logger.info("加载 FireRedVAD：%s（gpu=%s）", self.model_dir, self.use_gpu)
        self._vad = FireRedVad.from_pretrained(str(self.model_dir), config)

    def detect(self, wav: np.ndarray) -> list[tuple[float, float]]:
        """输入 16k int16 单声道数组，返回秒级 (start, end) 列表。"""
        self.load()
        result, _ = self._vad.detect(wav)
        segments = result.get("timestamps", []) if isinstance(result, dict) else []
        cleaned: list[tuple[float, float]] = []
        for start, end in segments:
            s, e = float(start), float(end)
            if e > s + 1e-3:
                cleaned.append((s, e))
        return cleaned
