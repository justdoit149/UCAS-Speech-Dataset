"""Qwen3-ASR 批量转写。"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)
MIN_INPUT_S = 0.6


def avg_logprob(scores, seqs, pad_id: int | None):
    """按生成 token 求平均 logprob，padding/结束符不计入。

    scores: [B, T, V] 每个解码步的 logits；seqs: [B, T] 生成的 token id。
    返回长度 B 的列表；某条没有可计分 token 时记 None。
    """
    import torch

    lp = torch.log_softmax(scores.float(), dim=-1).gather(2, seqs.unsqueeze(-1)).squeeze(-1)
    if pad_id is not None:
        mask = (seqs != pad_id).float()
    else:
        mask = torch.ones_like(seqs, dtype=torch.float)
    counts = mask.sum(1)
    vals = (lp * mask).sum(1) / counts.clamp(min=1)
    return [float(v) if c > 0 else None for v, c in zip(vals.tolist(), counts.tolist())]


class QwenASRAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_dir = cfg.model_dir("asr")
        self.device = cfg.device()
        self.dtype = cfg.torch_dtype()
        self.batch_size = int(cfg.get("asr.batch_size", 4))
        self.max_new_tokens = int(cfg.get("asr.max_new_tokens", 256))
        self.language = str(cfg.get("asr.language", "Chinese"))
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from qwen_asr import Qwen3ASRModel

        if not self.model_dir.exists():
            raise FileNotFoundError(f"ASR 模型目录不存在：{self.model_dir}")
        logger.info("加载 Qwen3-ASR：%s（device=%s, dtype=%s）",
                    self.model_dir, self.device, self.dtype)
        self._model = Qwen3ASRModel.from_pretrained(
            str(self.model_dir),
            dtype=self.dtype,
            device_map=self.device,
            max_inference_batch_size=max(1, self.batch_size),
            max_new_tokens=self.max_new_tokens,
        )

    def _pad_short(self, path: Path, tmpdir: Path) -> Path:
        info = sf.info(str(path))
        if info.duration >= MIN_INPUT_S:
            return path
        data, sr = sf.read(str(path), dtype="float32")
        need = int(round(MIN_INPUT_S * sr)) - len(data)
        if need > 0:
            data = np.pad(data, (0, need))
        out = tmpdir / f"{path.stem}-pad.wav"
        sf.write(str(out), data, sr, subtype="PCM_16")
        return out

    def _scored_transcribe(self, paths: list[str]) -> list[tuple[str, float | None]]:
        """直接调用底层 generate，拿到每个 token 的 logits 以计算置信度。"""
        import torch

        model = self._model
        wavs = [sf.read(p, dtype="float32")[0] for p in paths]
        prompts = [model._build_text_prompt("", self.language) for _ in paths]
        inputs = model.processor(text=prompts, audio=wavs, return_tensors="pt", padding=True)
        inputs = inputs.to(model.model.device).to(model.model.dtype)
        prompt_len = inputs["input_ids"].shape[1]
        gen = model.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, output_scores=True,
        )
        seqs = gen.sequences[:, prompt_len:]
        texts = model.processor.batch_decode(
            seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        if len(texts) != len(paths):
            raise RuntimeError(f"ASR 返回数量不符：{len(texts)} != {len(paths)}")
        scores = getattr(gen, "scores", None)
        pad_id = model.processor.tokenizer.pad_token_id
        confs = (
            avg_logprob(torch.stack(scores, dim=1), seqs, pad_id)
            if scores else [None] * len(paths)
        )
        return list(zip(texts, confs))

    def transcribe(self, wav_paths: list[str | Path]) -> list[tuple[str, float | None]]:
        self.load()
        with tempfile.TemporaryDirectory(prefix="voicedata_asr_") as td:
            tmpdir = Path(td)
            paths = [str(self._pad_short(Path(p), tmpdir)) for p in wav_paths]
            try:
                out = self._scored_transcribe(paths)
            except Exception as exc:  # 私有接口变动时不影响主流程，置信度记 None
                logger.warning("ASR 置信度计算失败，回退到普通转写：%s", exc)
                results = self._model.transcribe(
                    audio=paths,
                    language=[self.language] * len(paths),
                )
                out = []
                for item in results:
                    if hasattr(item, "text"):
                        text = item.text or ""
                    elif isinstance(item, dict):
                        text = str(item.get("text", ""))
                    else:
                        text = str(item)
                    out.append((text, None))
        if len(out) != len(wav_paths):
            raise RuntimeError(f"ASR 返回数量不符：{len(out)} != {len(wav_paths)}")
        return out
