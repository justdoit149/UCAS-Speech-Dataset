"""模型适配器工厂。"""

from __future__ import annotations

from .aed import FireRedAedAdapter
from .campplus import CampplusAdapter
from .fakes import FakeAED, FakeASR, FakeNoise, FakeSV, FakeVAD
from .firered import FireRedVADAdapter
from .noisy import NoisyAdapter
from .qwen import QwenASRAdapter


def build_adapters(cfg):
    if bool(cfg.get("run.fake_models", False)):
        return {
            "vad": FakeVAD(cfg), "aed": FakeAED(cfg),
            "sv": FakeSV(cfg), "asr": FakeASR(cfg), "noise": FakeNoise(cfg),
        }
    return {
        "vad": FireRedVADAdapter(cfg),
        "aed": FireRedAedAdapter(cfg),
        "sv": CampplusAdapter(cfg),
        "asr": QwenASRAdapter(cfg),
        "noise": NoisyAdapter(cfg),
    }
