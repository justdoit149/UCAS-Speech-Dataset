"""FireRedVAD AED：逐帧区分 speech / singing / music。"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FireRedAedAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_dir = cfg.model_dir("aed")
        self.use_gpu = cfg.device().startswith("cuda")
        self._aed = None

    def load(self) -> None:
        if self._aed is not None:
            return
        try:
            from fireredvad.aed import FireRedAed, FireRedAedConfig
        except ImportError:
            from fireredvad import FireRedAed, FireRedAedConfig
        if not (self.model_dir / "cmvn.ark").exists():
            raise FileNotFoundError(f"FireRedVAD AED 模型不完整，缺少 cmvn.ark：{self.model_dir}")
        if not (self.model_dir / "model.pth.tar").exists():
            raise FileNotFoundError(f"FireRedVAD AED 模型不完整，缺少 model.pth.tar：{self.model_dir}")
        config = FireRedAedConfig(
            use_gpu=self.use_gpu,
            smooth_window_size=5,
            speech_threshold=float(self.cfg.get("aed.speech_threshold", 0.4)),
            min_event_frame=int(float(self.cfg.get("aed.min_speech_ms", 300)) / 10),
            max_event_frame=2000,
            min_silence_frame=20,
            chunk_max_frame=30000,
        )
        logger.info("加载 FireRedVAD AED：%s（gpu=%s）", self.model_dir, self.use_gpu)
        self._aed = FireRedAed.from_pretrained(str(self.model_dir), config)

    def frame_probs(self, wav: np.ndarray) -> np.ndarray:
        """输入 16k int16 单声道数组，返回 (T,3) float32，列=[speech, singing, music]。"""
        self.load()
        _, probs = self._aed.detect(wav)
        arr = probs.numpy() if hasattr(probs, "numpy") else np.asarray(probs)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"AED 输出形状异常：{arr.shape}")
        return arr
