"""Qwen3-ASR 转写阶段（门槛低于导出阈值，便于以后调阈值）。"""

from __future__ import annotations

import json
import logging

from .batch import process_batch
from .db import Database
from .workcache import ensure_segment_audio

logger = logging.getLogger(__name__)


def _candidate_rows(db: Database, fid: str, vad_hash: str, params: str, gate: float):
    return db.query(
        "SELECT * FROM segments WHERE file_id=? AND vad_hash=? "
        "AND sv_state='done' AND sv_score >= ? "
        "AND (COALESCE(asr_state,'pending') != 'done' OR IFNULL(asr_params_hash,'') != ?) "
        "ORDER BY seq",
        (fid, vad_hash, gate, params),
    )


def run_asr(db: Database, cfg, job: str, file_ids: list[str], adapters, force: bool = False) -> None:
    params = cfg.stage_hash("asr", job)
    gate = float(cfg.get("sv.asr_min_score", 0.30))
    batch_size = int(cfg.get("asr.batch_size", 4))
    max_attempts = int(cfg.get("run.max_attempts", 3))
    backoff = float(cfg.get("run.retry_backoff_s", 1.0))

    for fid in file_ids:
        frow = db.get_file(fid)
        vad_stage = db.get_stage(fid, "vad")
        if frow is None or not vad_stage or vad_stage["status"] not in ("done", "partial"):
            if frow:
                logger.warning("跳过 ASR（VAD 未完成）：%s", frow["audio_file"])
            continue
        vad_hash = vad_stage["params_hash"]
        if force:
            db.execute(
                "UPDATE segments SET asr_state='pending' WHERE file_id=? AND vad_hash=? "
                "AND sv_state='done' AND sv_score >= ?",
                (fid, vad_hash, gate),
            )
            db.commit()
        rows = _candidate_rows(db, fid, vad_hash, params, gate)
        if not rows:
            if not db.stage_is_current(fid, "asr", params):
                db.begin_stage(fid, "asr", params)
                db.finish_stage(fid, "asr", "done", params,
                                stats=db.counts(fid, vad_hash, cfg.threshold()))
            continue

        db.begin_stage(fid, "asr", params)
        try:
            ensure_segment_audio(db, cfg, job, fid, rows)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("恢复片段音频失败：%s：%s", frow["audio_file"], msg)
            db.finish_stage(fid, "asr", "failed", params, error=msg)
            continue
        failed = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            db.executemany(
                "UPDATE segments SET asr_state='running' WHERE segment_id=?",
                [(r["segment_id"],) for r in batch],
            )
            db.commit()

            def infer(items):
                return adapters["asr"].transcribe([r["wav_path"] for r in items])

            ok, bad = process_batch(batch, infer, max_attempts, backoff)
            for row, result in ok:
                # 适配器返回 (text, avg_logprob)；假模型/旧接口可能只返回 text
                if isinstance(result, (tuple, list)) and len(result) == 2:
                    text, conf = result
                else:
                    text, conf = str(result), None
                raw = json.dumps({"text": text, "avg_logprob": conf}, ensure_ascii=False)
                db.update_asr(row["segment_id"], params, "done", text=text, raw=raw)
            for row, err in bad:
                failed += 1
                logger.error("ASR 失败：%s：%s", row["segment_id"], err)
                db.update_asr(row["segment_id"], params, "failed", text="", error=err)
            db.commit()
            done = min(start + len(batch), len(rows))
            logger.info("ASR 进度：%s %d/%d", frow["audio_file"], done, len(rows))

        stats = db.counts(fid, vad_hash, cfg.threshold())
        stats["failed"] = failed
        db.finish_stage(fid, "asr", "done" if failed == 0 else "partial", params, stats=stats)
        logger.info("ASR 完成：%s，%d 段，失败 %d", frow["audio_file"], len(rows), failed)
