"""声纹模板加载、余弦打分与声纹阶段。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .batch import process_batch
from .db import Database
from .workcache import ensure_segment_audio

logger = logging.getLogger(__name__)


def load_templates(path: str | Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 192:
        raise ValueError(f"声纹模板形状异常：{arr.shape}，应为 (N,192)")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if float(np.min(norms)) < 1e-6:
        raise ValueError("声纹模板中存在零向量")
    return arr / norms


def score_embeddings(embs: np.ndarray, templates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(embs, dtype=np.float32)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
    sims = e @ templates.T.astype(np.float32)
    idx = np.argmax(sims, axis=1)
    scores = sims[np.arange(len(idx)), idx]
    return scores.astype(np.float32), idx.astype(np.int32)


def run_sv(db: Database, cfg, job: str, file_ids: list[str], adapters, force: bool = False) -> None:
    params = cfg.stage_hash("sv", job)
    templates = load_templates(cfg.path("paths.template_path"))
    batch_size = int(cfg.get("sv.batch_size", 16))
    max_attempts = int(cfg.get("run.max_attempts", 3))
    backoff = float(cfg.get("run.retry_backoff_s", 1.0))

    for fid in file_ids:
        frow = db.get_file(fid)
        if frow is None:
            continue
        vad_stage = db.get_stage(fid, "vad")
        if not vad_stage or vad_stage["status"] not in ("done", "partial"):
            logger.warning("跳过声纹（VAD 未完成）：%s", frow["audio_file"])
            continue
        vad_hash = vad_stage["params_hash"]
        if force:
            db.execute(
                "UPDATE segments SET sv_state='pending' WHERE file_id=? AND vad_hash=?",
                (fid, vad_hash),
            )
            db.commit()
        rows = db.pending_segments(fid, "sv", params, vad_hash)
        if not rows:
            stats = db.counts(fid, vad_hash, cfg.threshold())
            if not db.stage_is_current(fid, "sv", params):
                db.begin_stage(fid, "sv", params)
                db.finish_stage(fid, "sv", "done", params, stats=stats)
            continue

        db.begin_stage(fid, "sv", params)
        try:
            ensure_segment_audio(db, cfg, job, fid, rows)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("恢复片段音频失败：%s：%s", frow["audio_file"], msg)
            db.finish_stage(fid, "sv", "failed", params, error=msg)
            continue
        failed = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            db.executemany(
                "UPDATE segments SET sv_state='running' WHERE segment_id=?",
                [(r["segment_id"],) for r in batch],
            )
            db.commit()

            def infer(items):
                paths = [r["wav_path"] for r in items]
                return adapters["sv"].embed(paths)

            ok, bad = process_batch(batch, infer, max_attempts, backoff)
            if ok:
                embs = np.stack([o for _, o in ok])
                scores, idxs = score_embeddings(embs, templates)
                for (row, emb), score, tidx in zip(ok, scores, idxs):
                    db.update_sv(
                        row["segment_id"], params, "done",
                        embedding=np.asarray(emb, dtype=np.float32).tobytes(),
                        score=float(score), template_idx=int(tidx),
                    )
                db.commit()
            for row, err in bad:
                failed += 1
                logger.error("声纹失败：%s：%s", row["segment_id"], err)
                db.update_sv(row["segment_id"], params, "failed", error=err)
            db.commit()
            done = min(start + len(batch), len(rows))
            logger.info("声纹进度：%s %d/%d", frow["audio_file"], done, len(rows))

        stats = db.counts(fid, vad_hash, cfg.threshold())
        stats["failed"] = failed
        db.finish_stage(fid, "sv", "done" if failed == 0 else "partial", params, stats=stats)
        logger.info("声纹完成：%s，%d 段，失败 %d", frow["audio_file"], len(rows), failed)
