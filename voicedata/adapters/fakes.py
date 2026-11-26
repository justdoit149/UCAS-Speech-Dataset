"""单元测试用的确定性假模型。"""

from __future__ import annotations

import numpy as np


class FakeVAD:
    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, wav: np.ndarray) -> list[tuple[float, float]]:
        dur = len(wav) / 16000.0
        out = []
        t = 0.5
        while t + 2.0 <= dur:
            out.append((t, t + 2.0))
            t += 3.0
        return out


class FakeAED:
    def __init__(self, cfg):
        self.cfg = cfg

    def frame_probs(self, wav: np.ndarray) -> np.ndarray:
        # 全帧 speech=1：切片门控不改变 FakeVAD 的分段，便于测试其余链路
        n = max(1, len(wav) // 160)
        return np.ones((n, 3), dtype=np.float32)


class FakeSV:
    def __init__(self, cfg):
        self.cfg = cfg

    def embed(self, wav_paths):
        # 固定方向，便于测试里用同一方向的模板得到 1.0 的余弦分
        vec = np.zeros(192, dtype=np.float32)
        vec[:96] = 1.0
        vec /= np.linalg.norm(vec)
        return np.stack([vec for _ in wav_paths])


class FakeASR:
    def __init__(self, cfg):
        self.cfg = cfg

    def transcribe(self, wav_paths):
        return [f"测试文本 {i}" for i in range(len(wav_paths))]


class FakeNoise:
    def __init__(self, cfg):
        self.cfg = cfg

    def classify(self, wav_paths):
        return [(0.9 if i % 2 == 0 else 0.1, 1 if i % 2 == 0 else 0) for i in range(len(wav_paths))]
