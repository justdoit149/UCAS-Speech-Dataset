"""Whisper 噪音/干净语音分类。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


class NoisyAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_dir = cfg.model_dir("noise")
        self.device = cfg.device()
        self.threshold = float(cfg.get("noise.threshold", 0.5))
        self._model = None
        self._extractor = None
        self._noisy_idx = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import WhisperFeatureExtractor, WhisperForAudioClassification

        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"噪音分类模型不完整：{self.model_dir}")
        logger.info("加载噪音分类器：%s（device=%s）", self.model_dir, self.device)
        self._extractor = WhisperFeatureExtractor.from_pretrained(str(self.model_dir))
        self._model = WhisperForAudioClassification.from_pretrained(
            str(self.model_dir), torch_dtype=torch.float32
        )
        self._model.eval()
        self._model.to(self.device if not self.device.startswith("cuda") else "cuda:0")
        cfg = self._model.config
        inv = {int(k): str(v) for k, v in cfg.id2label.items()}
        self._noisy_idx = next((i for i, name in inv.items() if name.lower() == "noisy"), 1)

    def classify(self, wav_paths: list[str | Path]) -> list[tuple[float, int]]:
        """返回 [(noisy_prob, flag)]。"""
        self.load()
        arrays = [sf.read(str(p), dtype="float32")[0] for p in wav_paths]
        inputs = self._extractor(
            arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        noisy_prob = probs[:, self._noisy_idx].cpu().numpy().astype(np.float32)
        return [(float(p), int(p >= self.threshold)) for p in noisy_prob]
