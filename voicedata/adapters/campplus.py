"""CAM++ 说话人 embedding 提取。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CampplusAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_dir = cfg.model_dir("sv")
        self.device = "gpu" if cfg.device().startswith("cuda") else "cpu"
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        from modelscope.pipelines import pipeline

        if not (self.model_dir / "configuration.json").exists():
            raise FileNotFoundError(f"CAM++ 模型不完整：{self.model_dir}")
        logger.info("加载 CAM++：%s（device=%s）", self.model_dir, self.device)
        self._pipeline = pipeline(
            task="speaker-verification",
            model=str(self.model_dir),
            device=self.device,
        )

    def embed(self, wav_paths: list[str | Path]) -> np.ndarray:
        """返回 L2 归一化后的 (N, 192) embedding。"""
        self.load()
        paths = [str(p) for p in wav_paths]
        out = self._pipeline(paths, output_emb=True)
        embs = np.asarray(out["embs"], dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-8)
