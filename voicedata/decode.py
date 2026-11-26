"""整文件解码为 16k/16bit/单声道 WAV。"""

from __future__ import annotations

import logging

from .audio import decode_to_wav, wav_info
from .db import Database

logger = logging.getLogger(__name__)


def run_decode(db: Database, cfg, job: str, file_ids: list[str], force: bool = False) -> None:
    sample_rate = int(cfg.get("decode.sample_rate", 16000))
    params = cfg.stage_hash("decode", job)
    work = cfg.job_dir("work", job)
    wav_dir = work / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)

    for fid in file_ids:
        row = db.get_file(fid)
        if row is None:
            continue
        wav_path = row["decoded_wav"]
        if (
            not force
            and row["decode_state"] == "done"
            and row["decode_params_hash"] == params
            and wav_path
            and row["decoded_duration_s"] is not None
        ):
            # 导出后 work 音频会被清理；这里只信状态库，避免每次重跑都重新解码。
            # 真正需要音频的阶段（vad / sv / asr / noise / export）会按需补齐。
            logger.debug("跳过解码（已完成）：%s", row["audio_file"])
            continue

        db.set_decode(fid, "running", params)
        db.begin_stage(fid, "decode", params)
        try:
            src = cfg.path("paths.input_dir") / row["rel_path"]
            offset = float(row["offset_s"] or 0.0)
            dur = float(row["excerpt_duration_s"]) if row["excerpt_duration_s"] else None
            out = wav_dir / f"{fid}.wav"
            decode_to_wav(src, out, sample_rate=sample_rate, offset_s=offset, duration_s=dur)
            info = wav_info(out)
            if info["sample_rate"] != sample_rate or info["channels"] != 1 or info["sample_width"] != 2:
                raise RuntimeError(f"解码结果格式异常：{info}")
            db.set_decode(fid, "done", params, str(out), info["duration_s"])
            db.finish_stage(fid, "decode", "done", params,
                            stats={"duration_s": round(info["duration_s"], 3)})
            logger.info("解码完成：%s（%.1fs）", row["audio_file"], info["duration_s"])
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("解码失败：%s：%s", row["audio_file"], msg)
            db.set_decode(fid, "failed", params, error=msg)
            db.finish_stage(fid, "decode", "failed", params, error=msg)
