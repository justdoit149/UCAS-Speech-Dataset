"""导出产物（zip/FLAC+JSONL 或兼容 JSON）的结构、音频与时间一致性校验。"""

from __future__ import annotations

import base64
import io
import json
import math
import wave
import zipfile
from pathlib import Path

import soundfile as sf

from .audio import sha256_file
from .config import Config

EXPECTED_AUDIO_FORMAT = "flac/16000/1/16bit"
REQUIRED_TOP = ("audio_file", "duration_s", "segment_count", "audio_format")
COMMON_SEG = ("segment_id", "start_s", "end_s", "asr_text", "noisy")
MAX_DUR_ERR_S = 0.05


def _check_fields(
    seg: dict, i: int, errors: list[str], total_src: float,
    prev_end: float, seen_ids: set[str],
) -> tuple[float, bool, bool]:
    """检查时间/文本/noisy/声纹等公共字段，返回 (prev_end, ok, 空文本)。"""
    lack = [k for k in COMMON_SEG if k not in seg]
    if lack:
        errors.append(f"段 {i}: 缺少字段 {lack}")
        return prev_end, False, False
    sid = seg.get("segment_id")
    if not isinstance(sid, str) or not sid:
        errors.append(f"段 {i}: segment_id 非法：{sid!r}")
    elif sid in seen_ids:
        errors.append(f"段 {i}: segment_id 重复：{sid}")
    else:
        seen_ids.add(sid)
    try:
        start, end = float(seg["start_s"]), float(seg["end_s"])
    except Exception:
        errors.append(f"段 {i}: 时间不是数字")
        return prev_end, False, False
    if not (math.isfinite(start) and math.isfinite(end)):
        errors.append(f"段 {i}: 时间非有限值")
        return prev_end, False, False
    if end <= start:
        errors.append(f"段 {i}: end_s({end}) <= start_s({start})")
        return prev_end, False, False
    if start < prev_end - 1e-3:
        errors.append(f"段 {i}: 与上一段重叠（{prev_end:.3f}s 之后又在 {start:.3f}s 开始）")
    prev_end = max(prev_end, end)
    if total_src and end > total_src + 1.0:
        errors.append(f"段 {i}: 结束时间 {end:.3f}s 超出音频时长 {total_src:.3f}s")
    text = seg.get("asr_text")
    empty = False
    if not isinstance(text, str):
        errors.append(f"段 {i}: asr_text 不是字符串")
    elif not text.strip():
        empty = True
    if seg.get("noisy") not in (0, 1, True, False):
        errors.append(f"段 {i}: noisy 应为 0/1，实际 {seg.get('noisy')!r}")
    if "sv_score" in seg:
        sc = seg["sv_score"]
        if not isinstance(sc, (int, float)) or not -1.0 <= float(sc) <= 1.0:
            errors.append(f"段 {i}: sv_score 异常：{sc!r}")
    return prev_end, True, empty


def _check_flac(raw: bytes, start: float, end: float, i: int, errors: list[str]) -> float:
    try:
        with sf.SoundFile(io.BytesIO(raw)) as f:
            ch, sr, subtype, n = f.channels, f.samplerate, f.subtype, len(f)
    except Exception as exc:
        errors.append(f"段 {i}: 不是合法 FLAC（{exc}）")
        return 0.0
    if (ch, sr) != (1, 16000):
        errors.append(f"段 {i}: FLAC 为 {ch}ch/{sr}Hz，应为 1/16000")
    if subtype != "PCM_16":
        errors.append(f"段 {i}: FLAC 子类型为 {subtype}，应为 PCM_16")
    dur = n / float(sr) if sr else 0.0
    if abs(dur - (end - start)) > MAX_DUR_ERR_S:
        errors.append(f"段 {i}: 音频 {dur:.3f}s 与标注 {end - start:.3f}s 差超过 {MAX_DUR_ERR_S}s")
    return dur


def _check_wav_b64(b64: str, start: float, end: float, i: int, errors: list[str]) -> float:
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:
        errors.append(f"段 {i}: base64 解码失败（{exc}）")
        return 0.0
    try:
        with wave.open(io.BytesIO(raw), "rb") as w:
            ch, sr, sw, n = w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes()
    except Exception as exc:
        errors.append(f"段 {i}: 不是合法 WAV（{exc}）")
        return 0.0
    if (ch, sr, sw) != (1, 16000, 2):
        errors.append(f"段 {i}: 音频格式为 {ch}ch/{sr}Hz/{sw * 8}bit，应为 1/16000/16")
    dur = n / float(sr) if sr else 0.0
    if abs(dur - (end - start)) > MAX_DUR_ERR_S:
        errors.append(f"段 {i}: 音频 {dur:.3f}s 与标注 {end - start:.3f}s 差超过 {MAX_DUR_ERR_S}s")
    return dur


def _finish_text_warnings(segs: list[dict], empty: int, warnings: list[str]) -> None:
    if segs and empty == len(segs):
        warnings.append("所有片段 asr_text 为空")
    elif empty:
        warnings.append(f"{empty}/{len(segs)} 段 asr_text 为空")


def validate_zip(path: str | Path, threshold: float | None = None) -> dict:
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    result = {"artifact": str(path), "format": "zip", "ok": False,
              "errors": errors, "warnings": warnings, "segments": 0, "speech_s": 0.0}
    try:
        zf = zipfile.ZipFile(path)
    except Exception as exc:
        errors.append(f"无法打开 zip：{exc}")
        return result
    with zf:
        try:
            doc = json.loads(zf.read("manifest.json").decode("utf-8"))
        except Exception as exc:
            errors.append(f"读取 manifest.json 失败：{exc}")
            return result
        missing = [k for k in REQUIRED_TOP if k not in doc]
        if missing:
            errors.append(f"缺少顶层字段：{missing}")
        if doc.get("format") != "flac-zip":
            warnings.append(f"manifest.format={doc.get('format')!r}，预期 flac-zip")
        if doc.get("audio_format") != EXPECTED_AUDIO_FORMAT:
            warnings.append(
                f"audio_format={doc.get('audio_format')!r}，预期 {EXPECTED_AUDIO_FORMAT}")
        try:
            body = zf.read("segments.jsonl").decode("utf-8")
        except Exception as exc:
            errors.append(f"读取 segments.jsonl 失败：{exc}")
            return result
        segs = []
        for n, line in enumerate(body.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                segs.append(json.loads(line))
            except ValueError as exc:
                errors.append(f"segments.jsonl 第 {n} 行解析失败：{exc}")
        if doc.get("segment_count") != len(segs):
            errors.append(f"segment_count={doc.get('segment_count')} 与实际 {len(segs)} 不符")
        if threshold is not None and float(doc.get("target_threshold", -1)) != float(threshold):
            warnings.append(
                f"target_threshold={doc.get('target_threshold')} 与 {threshold} 不同")

        total_src = float(doc.get("duration_s") or 0.0)
        prev_end = -1.0
        speech = 0.0
        empty_text = 0
        seen_ids: set[str] = set()
        for i, seg in enumerate(segs):
            prev_end, ok, empty = _check_fields(seg, i, errors, total_src, prev_end, seen_ids)
            if empty:
                empty_text += 1
            if not ok:
                continue
            audio = seg.get("audio")
            if not isinstance(audio, str) or not audio:
                errors.append(f"段 {i}: 缺少 audio 路径")
                continue
            if audio.startswith("/") or ".." in Path(audio).parts:
                errors.append(f"段 {i}: audio 路径不安全：{audio!r}")
                continue
            try:
                raw = zf.read(audio)
            except KeyError:
                errors.append(f"段 {i}: zip 中缺少音频成员 {audio}")
                continue
            except Exception as exc:
                errors.append(f"段 {i}: 读取音频失败：{exc}")
                continue
            speech += _check_flac(raw, float(seg["start_s"]), float(seg["end_s"]), i, errors)
        _finish_text_warnings(segs, empty_text, warnings)

    result.update({
        "ok": not errors,
        "segments": len(segs),
        "speech_s": round(speech, 3),
        "sha256": sha256_file(path),
    })
    return result


def validate_json(path: str | Path, threshold: float | None = None) -> dict:
    """兼容旧版单个 JSON（WAV base64 内嵌）。"""
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    result = {"artifact": str(path), "format": "json", "ok": False,
              "errors": errors, "warnings": warnings, "segments": 0, "speech_s": 0.0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as exc:
        errors.append(f"JSON 解析失败：{exc}")
        return result

    required = ("audio_file", "duration_s", "segment_count", "segments")
    missing = [k for k in required if k not in doc]
    if missing:
        errors.append(f"缺少顶层字段：{missing}")
    segs = doc.get("segments")
    if not isinstance(segs, list):
        errors.append("segments 不是数组")
        return result
    if doc.get("segment_count") != len(segs):
        errors.append(f"segment_count={doc.get('segment_count')} 与实际 {len(segs)} 不符")
    if threshold is not None and float(doc.get("target_threshold", -1)) != float(threshold):
        warnings.append(f"target_threshold={doc.get('target_threshold')} 与 {threshold} 不同")

    total_src = float(doc.get("duration_s") or 0.0)
    prev_end = -1.0
    speech = 0.0
    empty_text = 0
    seen_ids: set[str] = set()
    for i, seg in enumerate(segs):
        prev_end, ok, empty = _check_fields(seg, i, errors, total_src, prev_end, seen_ids)
        if empty:
            empty_text += 1
        if not ok:
            continue
        if "audio_base64" not in seg:
            errors.append(f"段 {i}: 缺少字段 ['audio_base64']")
            continue
        speech += _check_wav_b64(
            seg["audio_base64"], float(seg["start_s"]), float(seg["end_s"]), i, errors)
    _finish_text_warnings(segs, empty_text, warnings)

    result.update({
        "ok": not errors,
        "segments": len(segs),
        "speech_s": round(speech, 3),
        "sha256": sha256_file(path),
    })
    return result


def validate_artifact(path: str | Path, threshold: float | None = None) -> dict:
    suffix = Path(path).suffix.lower()
    if suffix == ".zip":
        return validate_zip(path, threshold)
    if suffix == ".json":
        return validate_json(path, threshold)
    return {
        "artifact": str(path), "format": suffix.lstrip("."), "ok": False,
        "errors": [f"不支持的产物类型：{suffix}"], "warnings": [],
        "segments": 0, "speech_s": 0.0,
    }


def _check_index(out_dir: Path, total_segments: int | None, total_files: int) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    index_path = out_dir / "index.jsonl"
    meta_path = out_dir / "index.meta.json"
    count = 0
    seen_ids: set[str] = set()
    if not index_path.exists():
        errors.append("缺少全局索引 index.jsonl")
    else:
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        seg = json.loads(line)
                    except ValueError as exc:
                        errors.append(f"index.jsonl 第 {n} 行解析失败：{exc}")
                        continue
                    lack = [k for k in ("audio_file", "archive", "segment_id") if k not in seg]
                    if lack:
                        errors.append(f"index.jsonl 第 {n} 行缺少字段 {lack}")
                        continue
                    if seg["segment_id"] in seen_ids:
                        errors.append(f"index.jsonl 第 {n} 行 segment_id 重复：{seg['segment_id']}")
                    seen_ids.add(seg["segment_id"])
                    if not (out_dir / seg["archive"]).exists():
                        errors.append(
                            f"index.jsonl 第 {n} 行引用的归档不存在：{seg['archive']}")
                    count += 1
        except Exception as exc:
            errors.append(f"读取 index.jsonl 失败：{exc}")
    if total_segments is not None and count != total_segments:
        errors.append(f"index.jsonl 共 {count} 段，与各 zip 合计 {total_segments} 段不符")

    if not meta_path.exists():
        errors.append("缺少 index.meta.json")
    else:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if total_segments is not None and meta.get("segment_count") != total_segments:
                errors.append(
                    f"index.meta.json segment_count={meta.get('segment_count')}，"
                    f"实际 {total_segments}")
            if meta.get("file_count") != total_files:
                errors.append(
                    f"index.meta.json file_count={meta.get('file_count')}，"
                    f"实际 {total_files}")
            if meta.get("audio_format") != EXPECTED_AUDIO_FORMAT:
                warnings.append(
                    f"index.meta.json audio_format={meta.get('audio_format')!r}，"
                    f"预期 {EXPECTED_AUDIO_FORMAT}")
        except Exception as exc:
            errors.append(f"读取 index.meta.json 失败：{exc}")
    return {"path": str(index_path), "ok": not errors, "segments": count,
            "file_count": total_files, "errors": errors, "warnings": warnings}


def validate_job(cfg: Config, job: str, file_pattern: str | None = None) -> dict:
    from .db import Database

    out_dir = cfg.job_dir("output", job)
    zips = sorted(out_dir.glob("*.zip"))
    if zips:
        files = zips
    else:
        # 旧 job 只有 JSON 时仍可校验，兼容模式不再生成索引
        files = sorted(out_dir.glob("*.json"))
    if file_pattern:
        files = [p for p in files if file_pattern.lower() in p.name.lower()]
    report = {"job": job, "files": [], "file_count": 0, "ok_count": 0,
              "total_segments": 0, "total_speech_s": 0.0,
              "index": None, "errors": [], "ok": True}
    for p in files:
        item = validate_artifact(p, threshold=None)
        report["files"].append(item)
        report["total_segments"] += item["segments"]
        report["total_speech_s"] += item["speech_s"]
        if item["ok"]:
            report["ok_count"] += 1
    report["file_count"] = len(files)
    report["total_speech_s"] = round(report["total_speech_s"], 3)

    # 与数据库记录交叉核对
    db_path = cfg.job_dir("work", job) / "state.db"
    if db_path.exists() and files:
        db = Database(db_path, readonly=True)
        try:
            for item, p in zip(report["files"], files):
                try:
                    row = db.one("SELECT * FROM exports WHERE artifact_path=?", (str(p),))
                except Exception:
                    # 尚未升级的旧库只有 json_path；只读连接不执行迁移
                    row = db.one("SELECT * FROM exports WHERE json_path=?", (str(p),))
                if row is None:
                    item["warnings"].append("数据库中没有对应导出记录")
                    continue
                keys = row.keys()
                sha = (row["artifact_sha256"] if "artifact_sha256" in keys
                       else row["json_sha256"])
                if sha and sha != item.get("sha256"):
                    item["errors"].append("产物 sha256 与数据库记录不一致（可能被手工改动）")
                if row["segment_count"] != item["segments"]:
                    item["errors"].append(
                        f"数据库记录段数 {row['segment_count']} 与文件 {item['segments']} 不符")
                item["ok"] = not item["errors"]
        finally:
            db.close()
        report["ok_count"] = sum(1 for i in report["files"] if i["ok"])

    if zips:
        index_total = None if file_pattern else report["total_segments"]
        report["index"] = _check_index(out_dir, index_total, len(zips))
        report["errors"].extend(report["index"]["errors"])
    else:
        report["index"] = {"path": None, "ok": True, "segments": 0,
                           "file_count": 0, "errors": [], "warnings": []}
    report["ok"] = (
        report["ok_count"] == report["file_count"] and not report["errors"]
    )
    return report
