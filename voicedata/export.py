"""导出最终数据集：zip（FLAC + JSONL）或兼容 JSON，并重建全局索引。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .audio import sha256_file, wav_base64, wav_to_flac_bytes
from .config import DEFAULT_FILLER_TEXTS
from .db import Database
from .workcache import cleanup_file_audio, ensure_segment_audio

FORMAT_ZIP = "zip"
FORMAT_JSON = "json"
ZIP_AUDIO_FORMAT = "flac/16000/1/16bit"
JSON_AUDIO_FORMAT = "wav/pcm_s16le/16000/1"

ZIP_MANIFEST = "manifest.json"
ZIP_SEGMENTS = "segments.jsonl"
ZIP_AUDIO_DIR = "audio"
INDEX_NAME = "index.jsonl"
INDEX_META_NAME = "index.meta.json"

# 归一化时去掉的空白与中英文标点
_PUNCT = set(
    "，。！？、；：…—·“”‘’《》〈〉【】「」『』（）()[]{}<>"
    "!?.,;:'\"-_/\\|@#$%^&*+=`~"
)
# 同一填充字重复（嗯嗯、好好好、对对对、是是是等）
_FILLER_RE = re.compile(r"^(?:嗯+|恩+|啊+|哦+|噢+|喔+|呃+|额+|唉+|诶+|哎+|喂+|好+|对+|是+|行+)$")


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in text.strip().lower() if not ch.isspace() and ch not in _PUNCT)


def _is_filler(text: str, fillers: set[str]) -> bool:
    norm = _normalize_text(text)
    if not norm:
        return False
    return norm in fillers or (len(norm) <= 4 and bool(_FILLER_RE.match(norm)))


def _asr_conf(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        doc = json.loads(raw)
    except (TypeError, ValueError):
        return None
    val = doc.get("avg_logprob") if isinstance(doc, dict) else None
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val)
    return None


def _skip_reason(
    row,
    *,
    drop_empty: bool,
    drop_fillers: bool,
    filler_max_s: float,
    fillers: set[str],
    min_conf: float | None,
) -> str | None:
    text = row["asr_text"] or ""
    if drop_empty and not text.strip():
        return "empty_text"
    duration = float(row["end_s"]) - float(row["start_s"])
    if drop_fillers and duration <= filler_max_s and _is_filler(text, fillers):
        return "filler"
    if min_conf is not None:
        conf = _asr_conf(row["asr_raw"])
        if conf is not None and conf < min_conf:
            return "low_conf"
    return None

logger = logging.getLogger(__name__)


def _eligible_rows(db: Database, fid: str, vad_hash: str, cfg, threshold: float):
    sv_h = cfg.stage_hash("sv", "")
    asr_h = cfg.stage_hash("asr", "")
    noise_h = cfg.stage_hash("noise", "")
    return db.query(
        "SELECT s.*, f.audio_file, f.stem, f.source_duration_s, f.offset_s, f.excerpt_duration_s "
        "FROM segments s JOIN files f ON f.file_id=s.file_id "
        "WHERE s.file_id=? AND s.vad_hash=? "
        "AND s.sv_state='done' AND s.sv_params_hash=? "
        "AND s.asr_state='done' AND s.asr_params_hash=? "
        "AND s.noise_state='done' AND s.noise_params_hash=? "
        "AND s.sv_score >= ? "
        "ORDER BY s.start_s",
        (fid, vad_hash, sv_h, asr_h, noise_h, threshold),
    )


class _HashingWriter:
    def __init__(self, f):
        self.f = f
        self.h = hashlib.sha256()

    def write(self, text: str) -> None:
        data = text.encode("utf-8")
        self.h.update(data)
        self.f.write(data)

    def hexdigest(self) -> str:
        return self.h.hexdigest()


def _build_meta(
    row, cfg, job: str, vad_hash: str, export_hash: str, threshold: float,
    schema_version: str, fmt: str, segment_count: int, generated_at: str,
) -> dict:
    meta = {
        "schema_version": schema_version,
        "format": "flac-zip" if fmt == FORMAT_ZIP else "json-base64",
        "audio_file": row["audio_file"],
        "duration_s": round(float(row["source_duration_s"] or 0.0), 3),
        "audio_format": ZIP_AUDIO_FORMAT if fmt == FORMAT_ZIP else JSON_AUDIO_FORMAT,
        "target_threshold": threshold,
        "segment_count": segment_count,
        "generated_at": generated_at,
        "models": {
            "vad": cfg.get("vad.model_name", "FireRedVAD"),
            "sv": cfg.get("sv.model_name", "campplus_sv_zh-cn_16k-common"),
            "asr": cfg.get("asr.model_name", cfg.get("asr.model_subdir")),
            "noise": cfg.get("noise.model_name", cfg.get("noise.model_subdir")),
        },
        "pipeline": {
            "job": job,
            "vad_hash": vad_hash,
            "sv_hash": cfg.stage_hash("sv", job),
            "asr_hash": cfg.stage_hash("asr", job),
            "noise_hash": cfg.stage_hash("noise", job),
            "export_hash": export_hash,
        },
    }
    if float(row["offset_s"] or 0.0) > 0 or row["excerpt_duration_s"]:
        meta["excerpt"] = {
            "offset_s": float(row["offset_s"] or 0.0),
            "duration_s": float(row["excerpt_duration_s"]) if row["excerpt_duration_s"] else None,
        }
    return meta


def _segment_base(row, text: str) -> dict:
    start = float(row["start_s"])
    end = float(row["end_s"])
    return {
        "segment_id": row["segment_id"],
        "start_s": round(start, 3),
        "end_s": round(end, 3),
        "duration_s": round(max(0.0, end - start), 3),
        "asr_text": text,
        "noisy": int(row["noisy_flag"] or 0),
    }


def _add_scores(seg: dict, row) -> None:
    seg["sv_score"] = round(float(row["sv_score"]), 4)
    seg["noise_prob"] = round(float(row["noisy_prob"] or 0.0), 4)
    conf = _asr_conf(row["asr_raw"])
    if conf is not None:
        seg["asr_conf"] = round(conf, 3)


def _zip_writestr(zf: zipfile.ZipFile, name: str, data: str | bytes) -> None:
    """固定时间戳 + 不压缩，保证同一份数据重复导出的内容稳定。"""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def _write_zip(out_path: Path, meta: dict, rows: list[tuple], include_scores: bool) -> tuple[str, float]:
    """写 <stem>.zip：manifest.json + audio/<id>.flac + segments.jsonl。"""
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    speech_s = 0.0
    lines: list[str] = []
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as zf:
            _zip_writestr(zf, ZIP_MANIFEST, json.dumps(meta, ensure_ascii=False))
            for row, text in rows:
                sid = row["segment_id"]
                member = f"{ZIP_AUDIO_DIR}/{sid}.flac"
                # 编码失败就让整个文件的导出失败，避免 manifest 段数对不上
                raw = wav_to_flac_bytes(row["wav_path"])
                _zip_writestr(zf, member, raw)
                seg = _segment_base(row, text)
                seg["audio"] = member
                if include_scores:
                    _add_scores(seg, row)
                lines.append(json.dumps(seg, ensure_ascii=False))
                speech_s += max(0.0, float(row["end_s"]) - float(row["start_s"]))
            payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
            _zip_writestr(zf, ZIP_SEGMENTS, payload)
        digest = sha256_file(tmp_path)
        os.replace(tmp_path, out_path)
        return digest, speech_s
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _write_json(out_path: Path, meta: dict, rows: list[tuple], include_scores: bool) -> tuple[str, float]:
    """兼容旧格式：单个 JSON，音频以 WAV base64 内嵌。"""
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    speech_s = 0.0
    try:
        with open(tmp_path, "wb") as f:
            w = _HashingWriter(f)
            w.write("{")
            first = True
            for key, value in meta.items():
                if not first:
                    w.write(",")
                first = False
                w.write(json.dumps(key, ensure_ascii=False) + ":")
                w.write(json.dumps(value, ensure_ascii=False))
            w.write(',"segments":[')
            prev_start = -1.0
            for i, (row, text) in enumerate(rows):
                start = float(row["start_s"])
                end = float(row["end_s"])
                if start < prev_start - 1e-6:
                    raise RuntimeError(f"片段顺序异常：{row['segment_id']}")
                prev_start = start
                seg = {
                    "segment_id": row["segment_id"],
                    "start_s": round(start, 3),
                    "end_s": round(end, 3),
                    "audio_base64": wav_base64(row["wav_path"]),
                    "asr_text": text,
                    "noisy": int(row["noisy_flag"] or 0),
                }
                if include_scores:
                    _add_scores(seg, row)
                if i:
                    w.write(",")
                w.write(json.dumps(seg, ensure_ascii=False))
                speech_s += max(0.0, end - start)
            w.write("]}")
            f.flush()
            os.fsync(f.fileno())
            digest = w.hexdigest()
        os.replace(tmp_path, out_path)
        return digest, speech_s
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _rebuild_index(out_dir: Path, cfg, job: str, schema_version: str) -> dict:
    """从 out 目录下所有 zip 的 segments.jsonl 重建全局索引。"""
    entries: list[tuple[str, float, str]] = []  # (audio_file, start_s, index line)
    file_count = 0
    segment_count = 0
    for zpath in sorted(out_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                manifest = json.loads(zf.read(ZIP_MANIFEST).decode("utf-8"))
                if manifest.get("format") != "flac-zip":
                    logger.warning("索引跳过非 FLAC-ZIP 产物：%s", zpath.name)
                    continue
                audio_file = manifest.get("audio_file", zpath.stem)
                body = zf.read(ZIP_SEGMENTS).decode("utf-8")
        except Exception as exc:
            logger.warning("索引跳过无法读取的 zip：%s：%s", zpath.name, exc)
            continue
        file_count += 1
        for raw in body.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                seg = json.loads(raw)
            except ValueError as exc:
                logger.warning("索引跳过无法解析的行：%s：%s", zpath.name, exc)
                continue
            line = {"audio_file": audio_file, "archive": zpath.name, **seg}
            try:
                start = float(seg.get("start_s", 0.0))
            except (TypeError, ValueError):
                start = 0.0
            entries.append((audio_file, start, json.dumps(line, ensure_ascii=False)))
            segment_count += 1

    # 同一录音内按时间升序；不同录音保持 zip 文件名顺序，避免把各录音的时间轴混在一起
    file_rank: dict[str, int] = {}
    for audio_file, _, _ in entries:
        file_rank.setdefault(audio_file, len(file_rank))
    entries.sort(key=lambda item: (file_rank[item[0]], item[1]))
    lines = [item[2] for item in entries]

    _write_text_atomic(out_dir / INDEX_NAME,
                       "\n".join(lines) + ("\n" if lines else ""))
    meta = {
        "schema_version": schema_version,
        "format": "flac-zip",
        "audio_format": ZIP_AUDIO_FORMAT,
        "job": job,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file_count": file_count,
        "segment_count": segment_count,
        "models": {
            "vad": cfg.get("vad.model_name", "FireRedVAD"),
            "sv": cfg.get("sv.model_name", "campplus_sv_zh-cn_16k-common"),
            "asr": cfg.get("asr.model_name", cfg.get("asr.model_subdir")),
            "noise": cfg.get("noise.model_name", cfg.get("noise.model_subdir")),
        },
    }
    _write_text_atomic(out_dir / INDEX_META_NAME,
                       json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    logger.info("索引已更新：%s（%d 个录音，%d 段）", INDEX_NAME, file_count, segment_count)
    return meta


def run_export(
    db: Database,
    cfg,
    job: str,
    file_ids: list[str],
    threshold_override: float | None = None,
    force: bool = False,
) -> None:
    threshold = cfg.threshold(threshold_override)
    export_hash = cfg.stage_hash("export", job, threshold)
    out_dir = cfg.job_dir("output", job)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = str(cfg.get("export.format", FORMAT_ZIP)).lower()
    if fmt not in (FORMAT_ZIP, FORMAT_JSON):
        raise ValueError(f"不支持的 export.format：{fmt}")
    drop_empty = bool(cfg.get("export.drop_empty_text", True))
    include_scores = bool(cfg.get("export.include_scores", True))
    schema_version = str(cfg.get("export.schema_version", "2.0"))
    drop_fillers = bool(cfg.get("export.drop_fillers", True))
    filler_max_s = float(cfg.get("export.filler_max_s", 1.2))
    fillers = {
        _normalize_text(str(x))
        for x in cfg.get("export.filler_texts", DEFAULT_FILLER_TEXTS)
    }
    fillers.discard("")
    min_conf_raw = cfg.get("export.min_asr_conf")
    min_conf = float(min_conf_raw) if min_conf_raw is not None else None
    cleanup_audio = bool(cfg.get("run.cleanup_audio_after_export", True))

    for fid in file_ids:
        frow = db.get_file(fid)
        if frow is None:
            continue
        vad_stage = db.get_stage(fid, "vad")
        if not vad_stage or vad_stage["status"] not in ("done", "partial"):
            logger.warning("跳过导出（VAD 未完成）：%s", frow["audio_file"])
            continue
        vad_hash = vad_stage["params_hash"]
        current = db.get_export(fid)
        if (
            not force
            and current
            and current["status"] == "done"
            and current["export_params_hash"] == export_hash
            and current["artifact_path"]
            and Path(current["artifact_path"]).exists()
        ):
            if cleanup_audio:
                try:
                    cleanup_file_audio(db, cfg, job, fid)
                except Exception as exc:
                    logger.warning("清理工作音频失败（不影响成品）：%s：%s",
                                   frow["audio_file"], exc)
            continue

        db.begin_stage(fid, "export", export_hash)
        rows = _eligible_rows(db, fid, vad_hash, cfg, threshold)
        candidate_count = len(rows)
        written_rows = []
        skipped = {"empty_text": 0, "filler": 0, "low_conf": 0, "missing_audio": 0}
        for row in rows:
            reason = _skip_reason(
                row, drop_empty=drop_empty, drop_fillers=drop_fillers,
                filler_max_s=filler_max_s, fillers=fillers, min_conf=min_conf,
            )
            if reason:
                skipped[reason] += 1
                continue
            text = row["asr_text"] or ""
            if not row["wav_path"]:
                logger.error("片段缺少音频路径，跳过：%s", row["segment_id"])
                skipped["missing_audio"] += 1
                continue
            # 清理后的 work 里没有片段 WAV；先收集，写 zip 前再统一按需恢复
            written_rows.append((row, text))

        # 输出必须按原音频时间升序，不依赖上游查询/过滤后的巧合顺序
        written_rows.sort(key=lambda item: (
            float(item[0]["start_s"]), float(item[0]["end_s"])))

        suffix = ".zip" if fmt == FORMAT_ZIP else ".json"
        out_path = out_dir / f"{frow['stem']}{suffix}"
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta = _build_meta(
            frow, cfg, job, vad_hash, export_hash, threshold,
            schema_version, fmt, len(written_rows), generated_at,
        )

        try:
            ensure_segment_audio(db, cfg, job, fid, written_rows)
            if fmt == FORMAT_ZIP:
                digest, speech_s = _write_zip(out_path, meta, written_rows, include_scores)
            else:
                digest, speech_s = _write_json(out_path, meta, written_rows, include_scores)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("导出失败：%s：%s", frow["audio_file"], msg)
            db.set_export({
                "file_id": fid, "status": "failed", "threshold": threshold,
                "export_params_hash": export_hash, "artifact_path": None,
                "artifact_sha256": None, "error": msg,
            })
            db.finish_stage(fid, "export", "failed", export_hash, error=msg)
            continue

        db.set_export({
            "file_id": fid,
            "status": "done",
            "artifact_path": str(out_path),
            "artifact_sha256": digest,
            "threshold": threshold,
            "export_params_hash": export_hash,
            "segment_count": len(written_rows),
            "skipped_count": max(0, candidate_count - len(written_rows)),
            "generated_at": generated_at,
        })
        db.finish_stage(fid, "export", "done", export_hash, stats={
            "format": fmt,
            "artifact": out_path.name,
            "segment_count": len(written_rows),
            "skipped_count": max(0, candidate_count - len(written_rows)),
            "skipped": skipped,
            "speech_s": round(speech_s, 2),
        })
        logger.info("导出完成：%s（%d 段，%.1fs 语音；跳过 空文本 %d、填充词 %d、低置信 %d、缺音频 %d）",
                    out_path.name, len(written_rows), speech_s,
                    skipped["empty_text"], skipped["filler"],
                    skipped["low_conf"], skipped["missing_audio"])
        if cleanup_audio:
            try:
                cleanup_file_audio(db, cfg, job, fid)
            except Exception as exc:
                logger.warning("清理工作音频失败（不影响成品）：%s：%s", frow["audio_file"], exc)

    if fmt == FORMAT_ZIP:
        try:
            _rebuild_index(out_dir, cfg, job, schema_version)
        except Exception as exc:
            # 单个文件都已落盘；索引失败时留下明确日志，validate 会再次报出来
            logger.error("全局索引生成失败：%s: %s", type(exc).__name__, exc)
    else:
        if (out_dir / INDEX_NAME).exists():
            logger.warning(
                "export.format=json 为兼容模式，不更新 index.jsonl；"
                "目录中的旧索引可能已过期，必要时请手工删除")
        else:
            logger.info("export.format=json 为兼容模式，不生成 index.jsonl")
